import os
import numpy as np
import pandas as pd
import onnxruntime as ort
import warnings

warnings.filterwarnings("ignore")

def process_csv_for_prediction(csv_path, input_length, window_step=1):
    """Process a CSV file using a sliding window to create samples for prediction"""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path, header=0, na_values=['NaN', 'nan'])
        
        required_cols = ['station', 'year', 'month', 'day', 'lon', 'lat', 'depth', 'temperature', 'salinity']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            print(f"File {csv_path} missing columns: {missing_cols}")
            return None, None

        # Convert columns to appropriate types
        for col in ['lon', 'lat', 'depth', 'temperature', 'salinity']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['station'] = df['station'].astype(str)
        
        # Group by timestamp and station
        groups = df.groupby(['year', 'month', 'day', 'station', 'lon', 'lat'])
        
        # Initialize lists for features and metadata
        all_features = []
        all_metadata = []
        depth_range = np.arange(10, 151)
        
        # Process each group (timestamp-station combination)
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
            all_features.append(feature_vector)
            all_metadata.append({
                'station': name[3],
                'year': name[0],
                'month': name[1],
                'day': name[2],
                'lon': name[4],
                'lat': name[5]
            })
        
        all_features = np.array(all_features, dtype=np.float32)
        
        if len(all_features) < input_length:
            print(f"CSV file contains {len(all_features)} timesteps, need at least {input_length}")
            return None, None

        # Create sliding window samples
        samples = []
        metadata_samples = []
        for i in range(0, len(all_features) - input_length + 1, window_step):
            sample = all_features[i:i + input_length]
            samples.append(sample)
            metadata_samples.append(all_metadata[i + input_length - 1])  # Store metadata of last timestep
        
        x_samples = np.array(samples, dtype=np.float32)
        x_samples = np.nan_to_num(x_samples, nan=np.nanmean(x_samples))
        
        return x_samples, metadata_samples
    
    except Exception as e:
        print(f"Error processing CSV file {csv_path}: {str(e)}")
        return None, None

def predict_from_csv(model_path, csv_path, window_step=1):
    """
    Predict using samples from a CSV file with a sliding window approach
    
    Parameters:
    model_path: Path to ONNX model file
    csv_path: Path to input CSV file
    window_step: Step size for sliding window
    """
    input_length = 24

    try:
        # Load ONNX model
        ort_session = ort.InferenceSession(model_path)
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    print(f"Processing CSV file: {csv_path}")
    
    X_test, metadata = process_csv_for_prediction(csv_path, input_length, window_step)

    if X_test is None:
        print(f"Failed to process CSV file {csv_path}")
        return

    # Run inference for all samples
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: X_test}
    
    ort_outs = ort_session.run(None, ort_inputs)
    y_pred_logits = ort_outs[0]
    
    # Process output
    y_pred_probs = 1 / (1 + np.exp(-y_pred_logits))  # Sigmoid
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # Print prediction results
    for i, pred in enumerate(y_pred):
        meta = metadata[i]
        print(f"Sample {i + 1}:")
        print(f"station: {meta['station']}, Date: {meta['year']}-{meta['month']}-{meta['day']}")
        print(f"Longitude: {meta['lon']:.4f}, Latitude: {meta['lat']:.4f}")
        print(f"Predicted Label: {pred}")
        print("-----------------")

    print("All samples processed")

if __name__ == '__main__':
    model_path = "models/ts_mixer_balanced_best.onnx"
    csv_path = "test_data/test_data.csv"  # Path to single CSV file
    window_step = 1  # Adjust step size as needed
    
    predict_from_csv(model_path, csv_path, window_step)