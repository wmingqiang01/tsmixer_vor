import os
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings

warnings.filterwarnings("ignore")

def process_single_file_for_prediction(file_path, input_length):
    """处理单个文件以进行预测"""
    try:
        df = pd.read_csv(file_path, delim_whitespace=True, header=0, na_values=['NaN', 'nan'])
        
        required_cols = ['station', 'year', 'month', 'day', 'lon', 'lat', 'depth', 'temperature', 'salinity', 'Label']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            print(f"文件 {file_path} 缺少列: {missing_cols}")
            return None, None, None
        
        for col in ['lon', 'lat', 'depth', 'temperature', 'salinity', 'Label']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['station'] = df['station'].astype(str)
        groups = df.groupby(['station', 'year', 'month', 'day', 'lon', 'lat', 'Label'])
        
        features_list = []
        labels_list = []
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
            labels_list.append(name[6])
            metadata_list.append({
                'station': name[0],
                'year': name[1],
                'month': name[2],
                'day': name[3],
                'lon': name[4],
                'lat': name[5]
            })
        
        features = np.array(features_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.float32)
        
        if len(features) > input_length:
            x_samples = np.lib.stride_tricks.sliding_window_view(
                features, (input_length, features.shape[1])
            ).reshape(-1, input_length, features.shape[1])
            y_samples = labels[input_length-1:-1]
            
            if len(x_samples) != len(y_samples):
                x_samples = x_samples[:-1]
            
            # 处理NaN
            x_samples = np.nan_to_num(x_samples, nan=np.nanmean(x_samples))
            
            return x_samples, y_samples, metadata_list[input_length-1:-1]
        
        return None, None, None
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return None, None, None

def main():
    onnx_model_path = 'models/ts_mixer_balanced_best.onnx'
    file_to_predict = 'data/test_data/test_data.txt' 
    input_length = 24

    print(f"Loading ONNX model from: {onnx_model_path}")
    try:
        ort_session = ort.InferenceSession(onnx_model_path)
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    print(f"\nProcessing data from: {file_to_predict}")
    X_test, y_true, metadata = process_single_file_for_prediction(file_to_predict, input_length)

    if X_test is None:
        print("No data produced after processing. Exiting.")
        return

    print(f"Data processed successfully. Shape of input for ONNX model: {X_test.shape}")

    # Run inference
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: X_test}
    
    print("\nRunning inference with ONNX Runtime...")
    ort_outs = ort_session.run(None, ort_inputs)
    y_pred_logits = ort_outs[0]
    
    # Post-process the output
    y_pred_probs = 1 / (1 + np.exp(-y_pred_logits)) # Sigmoid
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # Evaluate the results
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

    print("\n--- Prediction Results ---")
    print(f"File: {os.path.basename(file_to_predict)}")
    print(f"Total samples predicted: {len(y_pred)}")
    print(f"True positives (Label 1): {y_true.sum()}")
    print(f"Predicted positives (Label 1): {y_pred.sum()}")
    
    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("--------------------------\n")

    # Save predictions to CSV
    results_df = pd.DataFrame({
        'true_label': y_true,
        'predicted_prob': y_pred_probs,
        'predicted_label': y_pred
    })
    
    # Add metadata
    for key in metadata[0].keys():
        results_df[key] = [m[key] for m in metadata]
    
    output_file = 'predictions/predictions_onnx.csv'
    os.makedirs('predictions', exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"Predictions saved to: {output_file}")

if __name__ == '__main__':
    main()