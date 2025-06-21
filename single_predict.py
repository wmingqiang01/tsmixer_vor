import os
import numpy as np
import pandas as pd
import onnxruntime as ort
import warnings

warnings.filterwarnings("ignore")

def process_folder_for_prediction(folder_path, input_length):
    """处理一个文件夹（包含24个时间步的txt文件）以进行预测"""
    try:
        # 获取文件夹中所有txt文件
        txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
        
        if len(txt_files) != input_length:
            print(f"文件夹 {folder_path} 包含 {len(txt_files)} 个文件，期望 {input_length} 个文件")
            return None, None

        features_list = []
        metadata_list = []
        depth_range = np.arange(10, 151)

        # 按顺序处理每个txt文件
        for txt_file in txt_files:
            file_path = os.path.join(folder_path, txt_file)
            df = pd.read_csv(file_path, delim_whitespace=True, header=0, na_values=['NaN', 'nan'])
            
            required_cols = ['station', 'year', 'month', 'day', 'lon', 'lat', 'depth', 'temperature', 'salinity']
            if not all(col in df.columns for col in required_cols):
                missing_cols = [col for col in required_cols if col not in df.columns]
                print(f"文件 {file_path} 缺少列: {missing_cols}")
                return None, None
            
            for col in ['lon', 'lat', 'depth', 'temperature', 'salinity']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['station'] = df['station'].astype(str)
            groups = df.groupby(['station', 'year', 'month', 'day', 'lon', 'lat'])
            
            for name, group in groups:
                group = group.sort_values(by='depth')
                depth_values = group['depth'].values
                temp_values = group['temperature'].values
                sal_values = group['salinity'].values
                
                tem_sal_sequence = np.zeros(282)
                
                for i, depth in enumerate(depth_range):
                    mask = np.isclose(depth_values, depth, atol=1e-5)
                    if mask.sum() == 1:
                        idx = np.where(mask)[0][0]
                        tem_sal_sequence[i*2] = temp_values[idx]
                        tem_sal_sequence[i*2+1] = sal_values[idx]
                
                feature_vector = np.concatenate([[name[4], name[5]], tem_sal_sequence])
                features_list.append(feature_vector)
                metadata_list.append({
                    'station': name[0],
                    'year': name[1],
                    'month': name[2],
                    'day': name[3],
                    'lon': name[4],
                    'lat': name[5]
                })
        
        features = np.array(features_list, dtype=np.float32)
        
        if len(features) != input_length:
            print(f"文件夹 {folder_path} 处理后得到 {len(features)} 个时间步，期望 {input_length}")
            return None, None

        # 构造单个样本
        x_samples = features.reshape(1, input_length, features.shape[1])
        
        # 处理NaN
        x_samples = np.nan_to_num(x_samples, nan=np.nanmean(x_samples))
        
        return x_samples, metadata_list[-1:]  # 只返回最后一个时间步的元数据
    
    except Exception as e:
        print(f"处理文件夹 {folder_path} 时出错: {str(e)}")
        return None, None

def predict(model_path, data_path):
    onnx_model_path = model_path
    input_length = 24

    try:
        ort_session = ort.InferenceSession(onnx_model_path)
    except Exception as e:
        print(f"加载 ONNX 模型出错: {e}")
        return

    # 处理单个文件夹
    X_test, metadata = process_folder_for_prediction(data_path, input_length)

    if X_test is None:
        print("处理数据后无输出。退出程序。")
        return

    # 运行推理
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: X_test}
    
    ort_outs = ort_session.run(None, ort_inputs)
    y_pred_logits = ort_outs[0]
    
    # 处理输出
    y_pred_probs = 1 / (1 + np.exp(-y_pred_logits))  # Sigmoid
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # 打印预测结果
    for i, pred in enumerate(y_pred):
        meta = metadata[i]
        print(f"站号: {meta['station']}, 日期：{meta['year']}: {meta['month']}: {meta['day']}, "
              f"经度: {meta['lon']:.4f}, 纬度: {meta['lat']:.4f}, 预测标签: {pred}")
    print("-----------------")

if __name__ == '__main__':
    model_path = "models/ts_mixer_balanced_best.onnx"
    data_path = "test_data/samples/sample_0054"  # 改为文件夹路径
    predict(model_path, data_path)