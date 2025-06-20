import os
import numpy as np
import pandas as pd
import onnxruntime as ort
import warnings

warnings.filterwarnings("ignore")

def process_single_file_for_prediction(file_path, input_length):
    """处理单个文件以进行预测"""
    try:
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
        
        features_list = []
        metadata_list = []
        
        depth_range = np.arange(10, 151)
        
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
        
        if len(features) > input_length:
            x_samples = np.lib.stride_tricks.sliding_window_view(
                features, (input_length, features.shape[1])
            ).reshape(-1, input_length, features.shape[1])
            
            # 处理NaN
            x_samples = np.nan_to_num(x_samples, nan=np.nanmean(x_samples))
            
            return x_samples, metadata_list[input_length-1:]
        
        return None, None
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return None, None

def main(model_path, data_path):
    onnx_model_path = model_path
    file_to_predict = data_path
    input_length = 24

    try:
        ort_session = ort.InferenceSession(onnx_model_path)
    except Exception as e:
        print(f"加载 ONNX 模型出错: {e}")
        return

    X_test, metadata = process_single_file_for_prediction(file_to_predict, input_length)

    if X_test is None:
        print("处理数据后无输出。退出程序。")
        return

    # 运行推理
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: X_test}
    
    ort_outs = ort_session.run(None, ort_inputs)
    y_pred_logits = ort_outs[0]
    
    # 处理输出
    y_pred_probs = 1 / (1 + np.exp(-y_pred_logits)) # Sigmoid
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # 直接打印预测结果
    for i, pred in enumerate(y_pred):
        meta = metadata[i]
        print(f"站号: {meta['station']}, 日期：{meta['year']:}: {meta['month']}: {meta['day']}, "
              f"经度: {meta['lon']:.4f}, 纬度: {meta['lat']:.4f}, 预测标签: {pred}")
    print("-----------------")

if __name__ == '__main__':
    model_path = "models/ts_mixer_balanced_best.onnx"
    data_path = "test_data/test_data.txt"
    main(model_path, data_path)