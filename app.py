from flask import Flask, request, jsonify
from collections import deque
import threading
import time
import os
import base64
import uuid
import tempfile

import numpy as np
import matlab.engine

print("[Flask] Starting MATLAB engine...")
eng = matlab.engine.start_matlab()

MATLAB_CODE_DIR = r'E:\ECG_NEW'
MODEL_PATH      = r'E:\ECG_NEW\rfModel.mat'

eng.addpath(MATLAB_CODE_DIR, nargout=0)
eng.eval("rehash; clear functions; clear analyze_ecg_1min", nargout=0)

# 目标采样率（模型/特征/绘图都按这个）
FS_TARGET = 360
CAPTURE_SECONDS = 60

app = Flask(__name__)

capture_state = {
    "active": False,
    "start_time": None,
    "buffer": [],          # 一维 ECG 点列 [x0,x1,...]
    "status": "idle",      # idle / collecting / processing / done / error
    "result": None,
    "error": None,
    "capture_id": None
}

capture_lock = threading.Lock()
live_buffer = deque(maxlen=1000)  # 调试用：缓存最近若干包


# ----------------- 工具函数 -----------------

def _matlab_to_str(x):
    if x is None:
        return ""
    # 兼容 list/tuple/ndarray
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return ""
        return _matlab_to_str(x[0])
    return str(x)

def _mget(res, key, default=None):
    """兼容 matlab struct：res['x'] / getattr(res,'x') / res.get('x')"""
    try:
        return res[key]
    except Exception:
        pass
    try:
        return getattr(res, key)
    except Exception:
        pass
    try:
        return res.get(key, default)
    except Exception:
        return default

def _read_png_b64(path: str):
    if not path:
        return None
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def _cleanup_temp_png(*paths):
    """只删除 tempdir 下生成的图，避免删到你工程里的文件"""
    try:
        tmpdir = os.path.abspath(tempfile.gettempdir())
        for p in paths:
            if not p:
                continue
            pabs = os.path.abspath(p)
            if pabs.startswith(tmpdir) and os.path.exists(pabs):
                os.remove(pabs)
    except Exception:
        pass

def _resample_linear(x: np.ndarray, fs_in: float, fs_out: float):
    """简单线性重采样：把 fs_in 的序列重采样到 fs_out（不依赖 scipy）"""
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size < 2:
        return x
    if fs_in <= 1e-6 or fs_out <= 1e-6:
        return x

    # 如果差别很小，就不动
    if abs(fs_in - fs_out) / fs_out < 0.03:
        return x

    t_in = np.arange(x.size) / fs_in
    t_end = t_in[-1]
    n_out = int(np.floor(t_end * fs_out)) + 1
    if n_out < 2:
        return x

    t_out = np.arange(n_out) / fs_out
    y = np.interp(t_out, t_in, x)
    return y


# ----------------- MATLAB 分析 -----------------

def run_matlab_analysis(samples_flat, duration_sec: float):
    if not samples_flat:
        raise ValueError("no samples collected")

    ecg_raw = np.array(samples_flat, dtype=float).reshape(-1)
    n_raw = int(ecg_raw.size)

    # 用“实际收到点数 / 实际耗时”估计有效采样率
    if duration_sec and duration_sec > 1e-3:
        fs_est = float(n_raw / duration_sec)
    else:
        fs_est = float(FS_TARGET)

    # 关键：若 fs_est 偏离 360 太多，先重采样到 360
    ecg_for_matlab = ecg_raw
    fs_used = float(FS_TARGET)

    if abs(fs_est - FS_TARGET) / FS_TARGET > 0.10:
        ecg_for_matlab = _resample_linear(ecg_raw, fs_est, FS_TARGET)
        print(f"[Flask] fs_est={fs_est:.2f} != {FS_TARGET}, resample {n_raw} -> {ecg_for_matlab.size}")

    # MATLAB Engine 建议传列向量 N×1
    ecg_vec = matlab.double(ecg_for_matlab.reshape(-1, 1).tolist())

    print("[Flask] Calling MATLAB analyze_ecg_1min, samples:", int(ecg_for_matlab.size), "Fs=", fs_used)
    res = eng.analyze_ecg_1min(ecg_vec, fs_used, MODEL_PATH, nargout=1)

    # ---------- 结果字段 ----------
    labels_cell = _mget(res, 'labels', [])
    labels_py = [str(s) for s in labels_cell] if labels_cell is not None else []

    scores_mat = np.array(_mget(res, 'scores', []))
    scores_py = scores_mat.tolist() if scores_mat.size else []

    R_peaks_mat = np.array(_mget(res, 'R_peaks', [])).flatten()
    R_peaks_py = [int(x) for x in R_peaks_mat] if R_peaks_mat.size else []

    Q_onset_mat = np.array(_mget(res, 'Q_onset', [])).flatten()
    S_end_mat   = np.array(_mget(res, 'S_end', [])).flatten()
    Q_onset_py  = [int(x) for x in Q_onset_mat] if Q_onset_mat.size else []
    S_end_py    = [int(x) for x in S_end_mat] if S_end_mat.size else []

    # ---------- 两张图 ----------
    fig_path_1 = _matlab_to_str(_mget(res, 'rpeaks_fig_path', ''))
    fig_path_2 = _matlab_to_str(_mget(res, 'ecg_last10_fig_path', ''))

    img1_b64 = _read_png_b64(fig_path_1)
    img2_b64 = _read_png_b64(fig_path_2)

    if fig_path_1:
        print("[Flask] Figure#1:", fig_path_1, "ok" if img1_b64 else "missing")
    if fig_path_2:
        print("[Flask] Figure#2:", fig_path_2, "ok" if img2_b64 else "missing")

    _cleanup_temp_png(fig_path_1, fig_path_2)

    # ---------- 统计（沿用你的逻辑） ----------
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

        "rpeaks_image_base64": img1_b64,
        "last10_image_base64": img2_b64,

        # 调试信息：你可以在前端 JSON 看
        "n_samples_raw": n_raw,
        "duration_sec": float(duration_sec) if duration_sec else None,
        "fs_est": fs_est,
        "fs_used": fs_used,
        "n_samples_used": int(ecg_for_matlab.size),
    }


def _analysis_thread(samples_flat, duration_sec):
    global capture_state
    try:
        result = run_matlab_analysis(samples_flat, duration_sec)
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


# ----------------- 路由 -----------------

@app.route("/")
def index():
    return "ECG Flask backend is running."


@app.route("/start_capture", methods=["POST"])
def start_capture():
    with capture_lock:
        if capture_state["status"] in ("collecting", "processing"):
            return jsonify({"ok": False, "status": capture_state["status"], "msg": "capture already in progress"}), 400

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
        "capture_id": cid,
        "fs_target": FS_TARGET
    })


@app.route("/upload_data", methods=["POST"])
def upload_data():
    """
    ✅ 新推荐格式（你现在的“每包16点”）：
    1) 单包：
       {"capture_id":"xxx","samples":[...16个点...]}
    2) 批量：
       {"capture_id":"xxx","batch":[{"samples":[...16...]}, {"samples":[...16...]}, ...]}

    兼容旧格式（val1..val4）：
       {"capture_id":"xxx","val1":...,"val2":...,"val3":...,"val4":...}
       {"capture_id":"xxx","batch":[{"val1":...,"val2":...,"val3":...,"val4":...}, ...]}
    """
    data = request.get_json(silent=True) or {}
    req_cid = data.get("capture_id")

    with capture_lock:
        cur_status = capture_state["status"]
        cur_cid = capture_state.get("capture_id")
        collecting = (capture_state["active"] and cur_status == "collecting")

    if not collecting:
        return jsonify({"ok": False, "status": cur_status, "msg": "not collecting", "capture_id": cur_cid}), 409

    if not req_cid or req_cid != cur_cid:
        return jsonify({"ok": False, "status": cur_status, "msg": "capture_id mismatch", "capture_id": cur_cid}), 409

    # ---------- 解析 ----------
    samples_to_append = []

    def _append_samples(arr):
        nonlocal samples_to_append
        if not isinstance(arr, list):
            return
        for v in arr:
            try:
                samples_to_append.append(float(v))
            except Exception:
                samples_to_append.append(0.0)

    if isinstance(data.get("batch"), list):
        for item in data["batch"]:
            if isinstance(item, dict) and isinstance(item.get("samples"), list):
                _append_samples(item["samples"])
            else:
                # 兼容旧 val1..val4
                try:
                    v1 = float(item.get("val1", 0))
                    v2 = float(item.get("val2", 0))
                    v3 = float(item.get("val3", 0))
                    v4 = float(item.get("val4", 0))
                    samples_to_append.extend([v1, v2, v3, v4])
                except Exception:
                    continue
    else:
        # 单包：优先 samples
        if isinstance(data.get("samples"), list):
            _append_samples(data["samples"])
        else:
            # 兼容旧 val1..val4
            try:
                v1 = float(data.get("val1", 0))
                v2 = float(data.get("val2", 0))
                v3 = float(data.get("val3", 0))
                v4 = float(data.get("val4", 0))
                samples_to_append.extend([v1, v2, v3, v4])
            except Exception:
                return jsonify({"ok": False, "error": "invalid data"}), 400

    if not samples_to_append:
        return jsonify({"ok": False, "error": "empty samples"}), 400

    # ---------- 写入 ----------
    start_analysis = False
    samples_copy = None
    duration_sec = None

    with capture_lock:
        if capture_state["active"] and capture_state["status"] == "collecting" and capture_state["capture_id"] == req_cid:
            # 调试缓存：按“包”记一下（只记最后一段，别太大）
            live_buffer.append(samples_to_append[:16])

            capture_state["buffer"].extend(samples_to_append)

            elapsed = time.time() - capture_state["start_time"]
            n = len(capture_state["buffer"])

            # 只按时间结束（更符合“采 60 秒”）
            if elapsed >= CAPTURE_SECONDS:
                samples_copy = capture_state["buffer"][:]
                duration_sec = elapsed
                capture_state["active"] = False
                capture_state["status"] = "processing"
                start_analysis = True

            # 保险：如果点数异常暴涨（比如重复发送），也强制结束，避免内存爆
            elif n >= FS_TARGET * CAPTURE_SECONDS * 2:
                samples_copy = capture_state["buffer"][:FS_TARGET * CAPTURE_SECONDS * 2]
                duration_sec = elapsed
                capture_state["active"] = False
                capture_state["status"] = "processing"
                start_analysis = True

    if start_analysis and samples_copy is not None:
        t = threading.Thread(target=_analysis_thread, args=(samples_copy, duration_sec))
        t.daemon = True
        t.start()
        print("[Flask] Capture complete -> MATLAB analysis. samples:", len(samples_copy), "elapsed:", duration_sec)

    return jsonify({
        "ok": True,
        "accepted_points": len(samples_to_append)
    })


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
            progress = max(0.0, min(CAPTURE_SECONDS, elapsed))

    return jsonify({
        "status": status,
        "progress": progress,
        "result": result,
        "error": error,
        "capture_id": cid
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
