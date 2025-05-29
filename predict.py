import torch
import numpy as np
import pandas as pd
import os
import logging
import onnxruntime as ort
from dataset import VortexDataset
from torch.utils.data import DataLoader

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler('logs/predict.log', mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_onnx_model(model_path):
    """
    加载ONNX模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'{model_path} 文件不存在，请先训练模型')
    
    try:
        session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        return session
    except Exception as e:
        raise RuntimeError(f'加载ONNX模型失败: {str(e)}')

def predict_single(session, input_data):
    """
    对单个样本进行预测
    """
    # 确保输入是float32类型的numpy数组
    if isinstance(input_data, torch.Tensor):
        input_data = input_data.numpy()
    
    input_data = input_data.astype(np.float32)
    
    # 如果输入是单个样本，添加batch维度
    if len(input_data.shape) == 2:  # (seq_len, features)
        input_data = np.expand_dims(input_data, axis=0)  # (1, seq_len, features)
    
    # 执行推理
    outputs = session.run(None, {'input': input_data})[0]
    
    # 应用sigmoid并转换为二分类结果
    probs = 1 / (1 + np.exp(-outputs))  # sigmoid
    predictions = (probs >= 0.5).astype(int)
    
    return predictions, probs

def predict_batch(session, data_loader):
    """
    对批量数据进行预测
    """
    all_preds = []
    all_probs = []
    all_inputs = []
    
    for batch_idx, (x, y) in enumerate(data_loader):
        x_np = x.numpy().astype(np.float32)
        
        # ONNX推理
        outputs = session.run(None, {'input': x_np})[0]
        
        # 应用sigmoid并转换为二分类结果
        probs = 1 / (1 + np.exp(-outputs))
        preds = (probs >= 0.5).astype(int)
        
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_inputs.extend(x_np)
    
    return all_preds, all_probs, all_inputs

def main():
    logger = setup_logging()
    
    # 配置参数
    input_length = 8  # 输入序列长度
    batch_size = 32
    model_path = 'models/ts_mixer.onnx'
    data_dir = 'data/test_data'  # 预测数据目录
    output_dir = 'predictions'
    
    logger.info('开始预测:')
    logger.info(f'输入序列长度={input_length}, 批大小={batch_size}')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    session = load_onnx_model(model_path)
    logger.info(f'已加载ONNX模型: {model_path}')
    
    # 准备数据
    predict_dataset = VortexDataset(data_dir, input_length=input_length)
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f'预测样本数: {len(predict_dataset)}')
    
    # 执行预测
    predictions, probabilities, inputs = predict_batch(session, predict_loader)
    
    # 保存预测结果
    feature_cols = [f'Val{i}_{j}' for i in range(5, 123) for j in range(1, 3)]
    input_cols = [f'{col}_t{t+1}' for t in range(input_length) for col in feature_cols]
    
    output_data = []
    for i in range(len(predictions)):
        input_flat = inputs[i].reshape(-1)
        row = {
            **{col: input_flat[j] for j, col in enumerate(input_cols) if j < len(input_flat)},
            'prediction': int(predictions[i]),
            'probability': float(probabilities[i])
        }
        output_data.append(row)
    
    output_df = pd.DataFrame(output_data)
    output_path = os.path.join(output_dir, 'predictions1.csv')
    output_df.to_csv(output_path, index=False)
    logger.info(f'已保存预测结果到 {output_path}')
    
    # 统计预测结果
    positive_count = sum(predictions)
    logger.info(f'预测结果统计: 正例={positive_count}, 负例={len(predictions)-positive_count}')
    
    return predictions

if __name__ == '__main__':
    main()