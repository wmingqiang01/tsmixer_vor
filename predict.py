import torch
import numpy as np
import pandas as pd
import os
import onnxruntime as ort
from dataset import VortexDataset
import warnings
warnings.filterwarnings("ignore")

def load_onnx_model(model_path):
    """
    加载ONNX模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'{model_path} 文件不存在')
    
    try:
        session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        return session
    except Exception as e:
        raise RuntimeError(f'加载ONNX模型失败: {str(e)}')

def predict_single(session, input_data):
    """
    对单个样本进行预测
    """
    if isinstance(input_data, torch.Tensor):
        input_data = input_data.numpy()
    
    input_data = input_data.astype(np.float32)
    
    if len(input_data.shape) == 2:  # (seq_len, features)
        input_data = np.expand_dims(input_data, axis=0)  # (1, seq_len, features)
    
    outputs = session.run(None, {'input': input_data})[0]
    probs = 1 / (1 + np.exp(-outputs))  # sigmoid
    prediction = (probs >= 0.5).astype(int)
    
    return prediction[0], probs[0]

def main(data_file):
    # 配置
    input_length = 8
    model_path = 'models/ts_mixer.onnx'
      # 单个输入文件
    
    # 加载模型
    session = load_onnx_model(model_path)
    
    # 加载单个文件
    df = pd.read_csv(data_file, sep='\s+', header=0)
    
    # 获取特征列（与 VortexDataset 一致）
    feature_cols = [col for col in df.columns if col not in ['Track', 'Year', 'Month', 'Day', 'Label']][8:-200]
    if len(feature_cols) != 284:
        raise ValueError(f"预期 284 个特征列，实际得到 {len(feature_cols)} 个")
    
    # 提取特征
    features = df[feature_cols].values  # 形状: (n_rows, 284)
    features = np.nan_to_num(features, nan=np.nanmean(features, axis=0))
    
    # 确保文件有足够的行数
    if len(features) < input_length:
        raise ValueError(f"文件 {data_file} 的行数少于 {input_length}")
    
    # 取最后 input_length 行进行预测
    input_data = features[-input_length:]  # 形状: (input_length, 284)
    
    # 转换为 torch 张量
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    
    # 预测
    prediction, probability = predict_single(session, input_tensor)
    
    # 打印结果
    print(f"预测结果: {prediction}")

if __name__ == '__main__':
    data_file = 'data/test_data/file_10.txt'
    print('已处理文件: ',data_file)
    main(data_file)