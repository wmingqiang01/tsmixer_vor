import torch
import numpy as np
import pandas as pd
import os
import onnxruntime as ort
import glob
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

def process_itp_file(input_file_path):
    """
    处理单个ITP数据文件。
    按站点和日期分组，连接温度和盐度为序列。
    返回处理后的数据作为 DataFrame。
    """
    try:
        # 读取数据
        df = pd.read_csv(input_file_path, delim_whitespace=True, header=0,
                         names=['station', 'year', 'month', 'day', 'lon', 'lat', 'depth', 'temperature', 'salinity'],
                         na_values=['NaN', 'nan'])

        # 转换为数值
        for col in ['lon', 'lat', 'depth', 'temperature', 'salinity']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 按元数据分组
        df['station'] = df['station'].astype(str)
        grouped = df.groupby(['station', 'year', 'month', 'day', 'lon', 'lat'])

        # 处理数据
        processed_rows = []
        for name, group in grouped:
            group = group.sort_values(by='depth')
            tem_sal_sequence = []
            for _, row in group.iterrows():
                temp_val = row['temperature'] if pd.notna(row['temperature']) else np.nan
                sal_val = row['salinity'] if pd.notna(row['salinity']) else np.nan
                tem_sal_sequence.extend([temp_val, sal_val])
            
            # 构造行
            row = [name[0], name[1], name[2], name[3], name[4], name[5]] + tem_sal_sequence
            processed_rows.append(row)

        # 创建 DataFrame
        max_cols = max(len(row) for row in processed_rows)
        columns = ['Track', 'Year', 'Month', 'Day', 'Lon', 'Lat'] + \
                  [f'depth{i}_{j}' for i in range(6, (max_cols-6)//2 + 6) for j in [1, 2]]
        df = pd.DataFrame(processed_rows, columns=columns[:len(processed_rows[0])])
        
        return df

    except FileNotFoundError:
        print(f"错误: 输入文件 {input_file_path} 不存在")
        raise
    except pd.errors.EmptyDataError:
        print(f"错误: 输入文件 {input_file_path} 为空或格式不正确")
        raise
    except Exception as e:
        print(f"处理 {input_file_path} 时发生错误: {e}")
        raise



def main(input_length, model_path, data_dir):
    """
    主函数，处理多个输入文件并进行预测
    """
    # 加载模型
    session = load_onnx_model(model_path)
    
    # 获取所有输入文件
    file_list = glob.glob(os.path.join(data_dir, '*.csv'))
    if not file_list:
        raise ValueError(f"目录 {data_dir} 中未找到任何 .csv 文件")
    
    # 处理所有文件，合并数据
    all_dfs = []
    for file_path in file_list:
        df = process_itp_file(file_path)
        # df = process_inter1(df)  # Removed interpolation
        # df = process_inter2(df)  # Removed interpolation
        all_dfs.append(df)
    
    # 合并所有 DataFrame，按 Year, Month, Day 排序以确保时间顺序
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df[['Year', 'Month', 'Day']] = combined_df[['Year', 'Month', 'Day']].astype(int)
    combined_df = combined_df.sort_values(by=['Track']).reset_index(drop=True)
    
    # 提取特征
    feature_cols = ['Lon', 'Lat'] + [f'depth{i}_{j}' for i in range(10, 151) for j in [1, 2]]
    features = combined_df[feature_cols].values
    features = np.nan_to_num(features, nan=np.nanmean(features, axis=0))
    
    # 确保有足够的行数
    if len(features) < input_length:
        raise ValueError(f"合并后的数据行数不足，预期至少 {input_length} 行，实际 {len(features)}")
    
    # 确保特征数量正确
    expected_features = 284  # Lon, Lat + 141 depths * 2
    if features.shape[1] != expected_features:
        raise ValueError(f"特征数量错误，预期 {expected_features} 列，实际 {features.shape[1]} 列")
    
    # 取最后 input_length 行作为输入
    input_data = features[-input_length:]  # 形状: (input_length, 284)
    
    # 转换为 torch 张量
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    
    # 预测
    prediction, probability = predict_single(session, input_tensor)
    
    # 打印结果
    print(f"预测结果: {prediction}, 概率: {probability}")

if __name__ == '__main__':
    # 配置参数
    input_length = 8
    model_path = 'models/ts_mixer.onnx'
    data_dir = 'data/test_data/sample_10'
    
    main(input_length, model_path, data_dir)