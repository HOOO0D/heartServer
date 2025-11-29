# app.py
from flask import Flask, request, jsonify
from collections import deque
import threading
import time
import os
import base64

import numpy as np

# ==== MATLAB Engine ====
import matlab.engine

print("[Flask] Starting MATLAB engine...")
eng = matlab.engine.start_matlab()

# 把包含 analyze_ecg_1min.m 和模型的目录加到 MATLAB 路径里
MATLAB_CODE_DIR = r'E:\ECG_NEW'      # 这里放 analyze_ecg_1min.m
MODEL_PATH      = r'E:\ECG_NEW\rfModel.mat'

eng.addpath(MATLAB_CODE_DIR, nargout=0)

FS = 360.0   # 采样率

app = Flask(__name__)

# ========== 采集状态的全局变量 ==========

capture_state = {
    "active": False,       # 是否正在收集原始数据（true 表示正在收集 60s）
    "start_time": None,    # 开始采集的时间 time.time()
    "buffer": [],          # 已收集的数据：[[v1, v2, v3, v4], ...]
    "status": "idle",      # idle / collecting / processing / done / error
    "result": None,        # MATLAB 返回的结果（Python dict）
    "error": None          # 出错信息
}

capture_lock = threading.Lock()  # 并发访问保护

# 实时缓存（可选）
live_buffer = deque(maxlen=1000)  # 最近的数据


# ========== MATLAB 调用封装 ==========

def run_matlab_analysis(samples_4ch):
    """
    调用 MATLAB 对采集到的心电数据进行处理。
    samples_4ch: List[List[float]]，形状约为 (N, 4)，每行 [val1, val2, val3, val4]

    返回 Python dict，里面包含：
      - labels: [str, ...]
      - scores: [[p0, p1], ...]
      - R_peaks: [int, ...]
      - rpeaks_image_base64: "...."
      - 其他一些简单统计
    """
    if not samples_4ch:
        raise ValueError("no samples collected")

    arr = np.array(samples_4ch, dtype=float)  # (N,4)

    # 1. 选一个通道作为 ECG（例如 val1）
    #    如果你想用 val2，就改成 arr[:, 1]
    ecg_ch = arr[:, 0]

    # 2. 转成 matlab.double 列向量
    ecg_vec = matlab.double(ecg_ch.tolist())

    # 3. 调用 MATLAB 的 analyze_ecg_1min
    #    MATLAB 端函数签名:
    #      function result = analyze_ecg_1min(ecg_raw, Fs, modelPath)
    print("[Flask] Calling MATLAB analyze_ecg_1min, samples:", len(ecg_ch))
    res = eng.analyze_ecg_1min(ecg_vec, float(FS), MODEL_PATH, nargout=1)

    # 4. 从 MATLAB struct 解析字段
    #    注意：MATLAB Engine 返回的 struct 在 Python 里类似 dict，可以用 res['字段名']
    #    —— labels 是 cellstr（前面让你在 MATLAB 里 cellstr(labels)）
    labels_cell = res['labels']    # 这是 matlab 的 cell 数组
    labels_py = [str(s) for s in labels_cell]   # 转成 Python list[str]

    # scores 是 double 矩阵
    scores_mat = np.array(res['scores'])
    scores_py = scores_mat.tolist()

    # R_peaks 是列向量
    R_peaks_mat = np.array(res['R_peaks']).flatten()
    R_peaks_py = [int(x) for x in R_peaks_mat]

    # Q/S 如果你想要也可以传回前端
    Q_onset_mat = np.array(res['Q_onset']).flatten()
    S_end_mat   = np.array(res['S_end']).flatten()
    Q_onset_py  = [int(x) for x in Q_onset_mat]
    S_end_py    = [int(x) for x in S_end_mat]

    # R 峰示意图路径
    fig_path_raw = res['rpeaks_fig_path']
    # 兼容 cell/char 两种情况
    if isinstance(fig_path_raw, (list, tuple)):
        fig_path = str(fig_path_raw[0])
    else:
        fig_path = str(fig_path_raw)

    img_b64 = None
    if fig_path and os.path.exists(fig_path):
        with open(fig_path, 'rb') as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode('ascii')
        print("[Flask] R-peak figure loaded:", fig_path)
    else:
        print("[Flask] R-peak figure not found or empty:", fig_path)

    # 简单统计一下正常/异常比例（假设 Situation=0 正常，1 异常）
    # 如果你 labels 里就是 "0"/"1"，可以用下面逻辑；否则根据你的实际标签名改
    total_beats = len(labels_py)
    abnormal_beats = 0
    for lab in labels_py:
        lab_str = lab.strip()
        if lab_str in ('1', 'abnormal', 'Abnormal'):
            abnormal_beats += 1
    normal_beats = total_beats - abnormal_beats
    abnormal_ratio = (abnormal_beats / total_beats) if total_beats > 0 else 0.0

    return {
        "total_beats": total_beats,
        "abnormal_beats": abnormal_beats,
        "normal_beats": normal_beats,
        "abnormal_ratio": abnormal_ratio,
        "labels": labels_py,
        "scores": scores_py,
        "R_peaks": R_peaks_py,
        "Q_onset": Q_onset_py,
        "S_end": S_end_py,
        "rpeaks_image_base64": img_b64,
    }


def _analysis_thread(samples_4ch):
    """后台线程：调用 MATLAB 进行分析，并更新 capture_state。"""
    global capture_state
    try:
        result = run_matlab_analysis(samples_4ch)
        with capture_lock:
            capture_state["result"] = result
            capture_state["status"] = "done"
            capture_state["error"] = None
        print("[Flask] Analysis done")
    except Exception as e:
        with capture_lock:
            capture_state["result"] = None
            capture_state["status"] = "error"
            capture_state["error"] = str(e)
        print("[Flask] Analysis error:", e)


# ========== 路由 ==========

@app.route("/")
def index():
    return "ECG Flask backend is running."


@app.route("/upload_data", methods=["POST"])
def upload_data():
    """
    小程序持续 POST 数据的接口。
    预期 JSON 格式：
    {
      "val1": 123,
      "val2": 456,
      "val3": 789,
      "val4": 101
    }
    """
    data = request.get_json(silent=True) or {}
    try:
        val1 = float(data.get("val1", 0))
        val2 = float(data.get("val2", 0))
        val3 = float(data.get("val3", 0))
        val4 = float(data.get("val4", 0))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "invalid data"}), 400

    sample = [val1, val2, val3, val4]

    # 1. 实时缓存（可选）
    live_buffer.append(sample)

    # 2. 如果当前正在采集 60s，则把这条数据存进采集 buffer
    with capture_lock:
        if capture_state["active"] and capture_state["status"] == "collecting":
            capture_state["buffer"].append(sample)
            elapsed = time.time() - capture_state["start_time"]
            # 满足任一条件：时间 >= 60s 或 数据量 >= 21600
            if elapsed >= 60 or len(capture_state["buffer"]) >= 21600:
                # 采集结束，启动后台线程调用 MATLAB
                samples_copy = capture_state["buffer"][:]
                capture_state["active"] = False
                capture_state["status"] = "processing"

                t = threading.Thread(target=_analysis_thread, args=(samples_copy,))
                t.daemon = True
                t.start()
                print("[Flask] Capture complete, start MATLAB analysis, samples:", len(samples_copy))

    return jsonify({"ok": True})


@app.route("/start_capture", methods=["POST"])
def start_capture():
    """
    前端点击“开始采集 60s”时调用。
    服务端从现在开始收集之后 60s 的数据（由 /upload_data 推送进来）。
    """
    with capture_lock:
        if capture_state["status"] in ("collecting", "processing"):
            # 正在采集或处理，直接返回当前状态
            return jsonify({
                "ok": False,
                "status": capture_state["status"],
                "msg": "capture already in progress"
            }), 400

        # 重置状态，开始新的采集
        capture_state["active"] = True
        capture_state["start_time"] = time.time()
        capture_state["buffer"] = []
        capture_state["status"] = "collecting"
        capture_state["result"] = None
        capture_state["error"] = None

    print("[Flask] Capture started")
    return jsonify({
        "ok": True,
        "status": "collecting",
        "msg": "capture started, collecting next 60s data"
    })


@app.route("/capture_result", methods=["GET"])
def capture_result():
    """
    detect 页轮询这个接口，获取当前采集/分析状态和结果。
    返回示例：
    {
      "status": "collecting",  // 或 processing / done / idle / error
      "progress": 12.3,        // 如果在 collecting，表示已采集秒数（0~60）
      "result": {...},         // 仅在 done 时有值
      "error": null
    }
    """
    with capture_lock:
        status = capture_state["status"]
        result = capture_state["result"]
        error = capture_state["error"]
        progress = None

        if status == "collecting":
            elapsed = time.time() - capture_state["start_time"]
            progress = max(0.0, min(60.0, elapsed))

    return jsonify({
        "status": status,
        "progress": progress,
        "result": result,
        "error": error
    })


if __name__ == "__main__":
    # 用 python app.py 跑，不要再用 `flask run`，避免 debug 重启两次导致 MATLAB engine 启动两次
    # debug 可以先关掉，或者用 use_reloader=False
    app.run(host="0.0.0.0", port=5000, debug=False)
    # 或者：
    # app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
