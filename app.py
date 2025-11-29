# app.py
from flask import Flask, request, jsonify
from collections import deque
import threading
import time

# 如果要用 MATLAB Engine，取消下面两行注释，并确保已安装 matlab.engine
# import matlab.engine
# eng = matlab.engine.start_matlab()  # 建议在程序启动时起一次

app = Flask(__name__)

# ========== 采集状态的全局变量 ==========

capture_state = {
    "active": False,       # 是否正在收集原始数据（true 表示正在收集 60s）
    "start_time": None,    # 开始采集的时间 time.time()
    "buffer": [],          # 已收集的数据：[[v1, v2, v3, v4], ...]
    "status": "idle",      # idle / collecting / processing / done / error
    "result": None,        # MATLAB 返回的结果（任意结构）
    "error": None          # 出错信息
}

capture_lock = threading.Lock()  # 并发访问保护

# 你原来就有的“实时上传”数据缓存，如果需要可以继续用
live_buffer = deque(maxlen=1000)  # 用来保存最近的一些实时数据（可选）


# ========== MATLAB 调用封装 ==========

def run_matlab_analysis(samples_4ch):
    """
    调用 MATLAB 对采集到的心电数据进行处理。
    samples_4ch: List[List[float]]，形状约为 (N, 4)，每行 [val1, val2, val3, val4]

    返回值：可以是数字、字符串或 dict，取决于你的 MATLAB 函数。
    """
    # TODO: 根据你的实际 MATLAB 函数修改这里的调用方式
    # 下面是一个示例，假设你有 analyze_ecg.m:
    #
    # function result = analyze_ecg(data)
    #   % data 是 N x 4 的矩阵
    #   % 输出 result 可以是数字，也可以是 struct
    # end
    #
    # matlab_data = matlab.double(samples_4ch)  # 转成 N x 4 矩阵
    # result = eng.analyze_ecg(matlab_data, nargout=1)
    # return result

    # 为了示例先用一个“假分析”——计算每个通道的平均值
    # 真正使用时，只需要把上面注释的 MATLAB 调用打开即可
    import numpy as np
    arr = np.array(samples_4ch)  # (N,4)
    means = arr.mean(axis=0).tolist()
    # 返回一个 dict，前端展示会比较方便
    return {
        "channel_means": {
            "val1_mean": means[0],
            "val2_mean": means[1],
            "val3_mean": means[2],
            "val4_mean": means[3],
        },
        "sample_count": int(arr.shape[0])
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
    except Exception as e:
        with capture_lock:
            capture_state["result"] = None
            capture_state["status"] = "error"
            capture_state["error"] = str(e)


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

    # 1. 实时缓存（可选，用于监控或其它功能）
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

    return jsonify({
        "ok": True,
        "status": "collecting",
        "msg": "capture started, collecting next 60s data"
    })


@app.route("/capture_result", methods=["GET"])
def capture_result():
    """
    新页面轮询这个接口，获取当前采集/分析状态和结果。
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
    # 这里 host 要写成 0.0.0.0，方便手机 / 小程序通过你电脑 IP 访问
    app.run(host="0.0.0.0", port=5000, debug=True)
