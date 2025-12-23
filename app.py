# app.py
from flask import Flask, request, jsonify
from collections import deque
import threading
import time
import os
import base64
import uuid

import numpy as np
import matlab.engine

print("[Flask] Starting MATLAB engine...")
eng = matlab.engine.start_matlab()

MATLAB_CODE_DIR = r'E:\ECG_NEW'
MODEL_PATH      = r'E:\ECG_NEW\rfModel.mat'

eng.addpath(MATLAB_CODE_DIR, nargout=0)
eng.eval("rehash; clear functions; clear analyze_ecg_1min", nargout=0)

# 硬件真实采样率：单片机 360Hz，每4点打包；后端展开后仍按360Hz
FS = 360
#FS=112

app = Flask(__name__)

capture_state = {
    "active": False,
    "start_time": None,
    "buffer": [],          # 一维 ECG 点列 [x0,x1,...]
    "status": "idle",      # idle / collecting / processing / done / error
    "result": None,
    "error": None,
    "capture_id": None     #  本轮采集会话ID
}

capture_lock = threading.Lock()

# 可选：调试缓存（只在 collecting 且 capture_id 匹配时写入）
live_buffer = deque(maxlen=1000)


def _matlab_to_str(x):
    """把 MATLAB Engine 返回的字符串/char/list/tuple/ndarray 尽量稳地转成 Python str"""
    if x is None:
        return ""

    # MATLAB 有时会返回 matlab.mlarray.char（看版本）
    try:
        import matlab  # matlab.engine 已经在用，一般也能 import matlab
        if isinstance(x, matlab.mlarray.char):
            # matlab char -> python str
            return ''.join(x)
    except Exception:
        pass

    # 兼容 list/tuple/ndarray：常见是 ['C:\\...png'] 或 array(['C:\\...'], dtype=object)
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return ""
        return _matlab_to_str(x[0])

    # 兜底
    return str(x)


def _img_to_b64(path, cleanup_temp=True):
    """读取 PNG 并转 base64；可选清理 tempdir 下的临时文件"""
    if not path:
        return None
    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        b = f.read()
    b64 = base64.b64encode(b).decode("ascii")

    # 可选：清理临时目录下的文件，避免越跑越多
    if cleanup_temp:
        try:
            tmp = os.path.abspath(tempfile.gettempdir())
            pabs = os.path.abspath(path)
            if pabs.startswith(tmp):
                os.remove(pabs)
        except Exception:
            pass

    return b64


def run_matlab_analysis(samples_flat):
    if not samples_flat:
        raise ValueError("no samples collected")

    ecg_ch = np.array(samples_flat, dtype=float).reshape(-1)

    # 建议传列向量 N×1，减少形状歧义
    ecg_vec = matlab.double(ecg_ch.reshape(-1, 1).tolist())

    print("[Flask] Calling MATLAB analyze_ecg_1min, samples:", len(ecg_ch))
    res = eng.analyze_ecg_1min(ecg_vec, float(FS), MODEL_PATH, nargout=1)

    # ---------- 1) 分类结果 ----------
    labels_cell = res.get('labels', [])
    labels_py = [str(s) for s in labels_cell]

    scores_mat = np.array(res.get('scores', []))
    scores_py = scores_mat.tolist() if scores_mat.size else []

    R_peaks_mat = np.array(res.get('R_peaks', [])).flatten()
    R_peaks_py = [int(x) for x in R_peaks_mat] if R_peaks_mat.size else []

    Q_onset_mat = np.array(res.get('Q_onset', [])).flatten()
    S_end_mat   = np.array(res.get('S_end', [])).flatten()
    Q_onset_py  = [int(x) for x in Q_onset_mat] if Q_onset_mat.size else []
    S_end_py    = [int(x) for x in S_end_mat] if S_end_mat.size else []

    # ---------- 2) 两张图：路径 -> base64 ----------
    # 第一张：原来的（例如前10秒+R峰）
    fig_path_1 = _matlab_to_str(res.get('rpeaks_fig_path', ''))
    img1_b64 = None
    if fig_path_1 and os.path.exists(fig_path_1):
        with open(fig_path_1, 'rb') as f:
            img1_b64 = base64.b64encode(f.read()).decode('ascii')
        print("[Flask] Figure#1 loaded:", fig_path_1)
    else:
        print("[Flask] Figure#1 not found or empty:", fig_path_1)

    # 第二张：最后10秒 ECG 图（你新增的字段）
    fig_path_2 = _matlab_to_str(res.get('ecg_last10_fig_path', ''))
    img2_b64 = None
    if fig_path_2 and os.path.exists(fig_path_2):
        with open(fig_path_2, 'rb') as f:
            img2_b64 = base64.b64encode(f.read()).decode('ascii')
        print("[Flask] Figure#2 loaded:", fig_path_2)
    else:
        print("[Flask] Figure#2 not found or empty:", fig_path_2)

    # （可选）编码后删除临时文件，防止 tempdir 越积越多
    # 注意：如果你想保留文件用于调试，就注释掉这段
    try:
        import tempfile
        tmpdir = os.path.abspath(tempfile.gettempdir())
        for p in (fig_path_1, fig_path_2):
            if p:
                pabs = os.path.abspath(p)
                if pabs.startswith(tmpdir) and os.path.exists(pabs):
                    os.remove(pabs)
    except Exception:
        pass

    # ---------- 3) 统计异常比例（保留你原逻辑） ----------
    total_beats = len(labels_py)
    abnormal_beats = 0
    normal_beats = 0
    for lab in labels_py:
        lab_str = str(lab).strip()
        if lab_str in ('1', 'abnormal', 'Abnormal'):
            abnormal_beats += 1
        else:
            normal_beats += 1

    abnormal_ratio = (abnormal_beats / total_beats) if total_beats > 0 else 0.0

    if normal_beats > 200:
        labels_py = ['abnormal' for _ in labels_py]
        abnormal_beats = total_beats
        normal_beats = 0
        abnormal_ratio = 1.0

    # ---------- 4) 返回结构：新增第二张图 ----------
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

        # 第一张：原来的图（保持兼容字段名）
        "rpeaks_image_base64": img1_b64,

        # ✅ 第二张：最后10秒图（新增）
        "last10_image_base64": img2_b64,

        # （可选）带上路径便于你调试；前端不需要就删掉
        "rpeaks_fig_path": fig_path_1,
        "last10_fig_path": fig_path_2,
    }



def _analysis_thread(samples_flat):
    global capture_state
    try:
        result = run_matlab_analysis(samples_flat)
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


@app.route("/")
def index():
    return "ECG Flask backend is running."


@app.route("/start_capture", methods=["POST"])
def start_capture():
    with capture_lock:
        if capture_state["status"] in ("collecting", "processing"):
            return jsonify({
                "ok": False,
                "status": capture_state["status"],
                "msg": "capture already in progress"
            }), 400

        cid = uuid.uuid4().hex

        capture_state["active"] = True
        capture_state["start_time"] = time.time()
        capture_state["buffer"] = []
        capture_state["status"] = "collecting"
        capture_state["result"] = None
        capture_state["error"] = None
        capture_state["capture_id"] = cid

    print("[Flask] Capture started, capture_id =", cid)
    return jsonify({
        "ok": True,
        "status": "collecting",
        "msg": "capture started, collecting next 60s data",
        "capture_id": cid
    })


@app.route("/upload_data", methods=["POST"])
def upload_data():
    """
    支持两种格式：
    1) 单包：
       {"capture_id":"xxx","val1":...,"val2":...,"val3":...,"val4":...}
    2) 批量：
       {"capture_id":"xxx","batch":[{"val1":...,"val2":...,"val3":...,"val4":...}, ...]}
    """
    data = request.get_json(silent=True) or {}
    req_cid = data.get("capture_id")

    with capture_lock:
        cur_status = capture_state["status"]
        cur_cid = capture_state.get("capture_id")
        collecting = (capture_state["active"] and cur_status == "collecting")

    # 1) 非 collecting：快速返回，帮助前端止血（避免尾流占用）
    if not collecting:
        return jsonify({
            "ok": False,
            "status": cur_status,
            "msg": "not collecting",
            "capture_id": cur_cid
        }), 409

    # 2) capture_id 不匹配：拒绝写入（防止尾流污染下一轮）
    if not req_cid or req_cid != cur_cid:
        return jsonify({
            "ok": False,
            "status": cur_status,
            "msg": "capture_id mismatch",
            "capture_id": cur_cid
        }), 409

    # 3) 解析数据
    packets = []

    if isinstance(data.get("batch"), list):
        for item in data["batch"]:
            try:
                v1 = float(item.get("val1", 0))
                v2 = float(item.get("val2", 0))
                v3 = float(item.get("val3", 0))
                v4 = float(item.get("val4", 0))
                packets.append([v1, v2, v3, v4])
            except (TypeError, ValueError):
                continue
    else:
        try:
            v1 = float(data.get("val1", 0))
            v2 = float(data.get("val2", 0))
            v3 = float(data.get("val3", 0))
            v4 = float(data.get("val4", 0))
            packets.append([v1, v2, v3, v4])
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "invalid data"}), 400

    if not packets:
        return jsonify({"ok": False, "error": "empty packets"}), 400

    # 4) 写入采集 buffer
    with capture_lock:
        # 再次确认状态（双检更稳）
        if capture_state["active"] and capture_state["status"] == "collecting" and capture_state["capture_id"] == req_cid:
            for packet in packets:
                # 可选：调试缓存
                live_buffer.append(packet)
                # 展开写入
                capture_state["buffer"].extend(packet)

            elapsed = time.time() - capture_state["start_time"]

            if elapsed >= 60 or len(capture_state["buffer"]) >= FS * 60:
                samples_copy = capture_state["buffer"][:]
                capture_state["active"] = False
                capture_state["status"] = "processing"

                t = threading.Thread(target=_analysis_thread, args=(samples_copy,))
                t.daemon = True
                t.start()
                print("[Flask] Capture complete, start MATLAB analysis, samples:", len(samples_copy))

    return jsonify({"ok": True, "accepted_packets": len(packets)})


@app.route("/capture_result", methods=["GET"])
def capture_result():
    with capture_lock:
        status = capture_state["status"]
        result = capture_state["result"]
        error = capture_state["error"]
        cid = capture_state.get("capture_id")
        progress = None

        if status == "collecting":
            elapsed = time.time() - capture_state["start_time"]
            progress = max(0.0, min(60.0, elapsed))

    return jsonify({
        "status": status,
        "progress": progress,
        "result": result,
        "error": error,
        "capture_id": cid
    })


if __name__ == "__main__":
    # 建议直接 python app.py 跑，避免 reloader 导致 MATLAB engine 启动两次
    app.run(host="0.0.0.0", port=5000, debug=False)
