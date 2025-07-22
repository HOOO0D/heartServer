from flask import Flask, request, jsonify
import matlab.engine
import threading
import os
app = Flask(__name__)


# 获取当前 Python 脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
eng = matlab.engine.start_matlab()
# 把当前目录加到 MATLAB 路径
eng.addpath(current_dir, nargout=0)
eng.addpath("D:\\ECG_info_RandomForest")
# 存储所有 86400 个点的一维 list
data_list = []

MAX_LENGTH = 800  # 最终目标点数

@app.route('/upload_data', methods=['POST'])
def upload_data():
    global data_list

    data = request.get_json()

    # 依次加入 val1~val4 到列表
    data_list.extend([
        float(data.get('val1', 0)),
        float(data.get('val2', 0)),
        float(data.get('val3', 0)),
        float(data.get('val4', 0)),
    ])
    print(f"getDATA: %f,%f,%f,%f",float(data.get('val1', 0)),
        float(data.get('val2', 0)),
        float(data.get('val3', 0)),
        float(data.get('val4', 0)))
    if len(data_list) >= MAX_LENGTH:
        # 数据满了，开始处理
        del data_list[0:MAX_LENGTH]
        threading.Thread(target=process_data).start()


    return jsonify({'status': 'ok', 'current_len': len(data_list)})


def process_data():
    global data_list

    try:
        print(f"开始处理数据，总点数: {len(data_list)}")

        # 转成 MATLAB 列向量格式：[[x1], [x2], ...]
        matlab_data = matlab.double([[x] for x in data_list])  # 86400x1

        result = eng.process_ecg(matlab_data)

        print("预测完成，结果为：", result)

        # 清空缓存
        # data_list = []

    except Exception as e:
        print("处理数据失败：", str(e))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)