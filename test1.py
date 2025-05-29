import torch
from torch.utils.data import DataLoader
from dataset import VortexDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import os
import logging
import numpy as np
from sklearn.model_selection import train_test_split
import onnxruntime as ort

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler('logs/test.log', mode='w'),
            logging.StreamHandler()
        ]
    )

def test():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    input_length = 10
    no_feats = 282
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info('Starting testing with hyperparameters:')
    logger.info(f'input_length={input_length}, no_feats={no_feats}, batch_size={batch_size}')
    
    # Data
    data_dir = 'data/output_data'
    test_dataset = VortexDataset(data_dir, split='test', input_length=input_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    logger.info(f'Test samples: {len(test_dataset)}')
    expected_batches = (len(test_dataset) + batch_size - 1) // batch_size
    logger.info(f'Expected batches: {expected_batches}')
    
    # Load ONNX model
    onnx_path = 'models/ts_mixer.onnx'
    if not os.path.exists(onnx_path):
        logger.error(f'ONNX model file {onnx_path} not found. Please train the model first.')
        raise FileNotFoundError(f'{onnx_path} not found')
    
    try:
        session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        logger.info(f'Loaded ONNX model from {onnx_path}')
    except Exception as e:
        logger.error(f'Failed to load ONNX model: {str(e)}')
        raise
    
    # Test
    criterion = torch.nn.BCEWithLogitsLoss()
    test_loss = 0.0
    test_preds, test_labels = [], []
    all_inputs = []
    all_metadata = []
    
    # Metadata
    feature_cols = [f'Val{i}_{j}' for i in range(5, 123) for j in range(1, 3)]
    data_files = test_dataset.data_files
    logger.info(f'Data files: {data_files}')
    all_features = []
    all_labels = []
    all_rows = []
    
    for file_path in data_files:
        df = pd.read_csv(file_path, sep='\s+', header=0)
        features = df[feature_cols].values
        labels = df['Label'].values
        for i in range(len(features) - input_length):
            x = features[i:i+input_length]
            y = labels[i+input_length]
            row = df.iloc[i+input_length][['Track', 'Year', 'Month', 'Day']].to_dict()
            all_features.append(x)
            all_labels.append(y)
            all_rows.append(row)
    
    logger.info(f'Total samples before split: {len(all_features)}')
    
    # Stratified split indices
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        range(len(all_features)), all_labels,
        test_size=0.3, stratify=all_labels, random_state=42
    )
    val_idx, test_idx, val_labels, test_labels = train_test_split(
        temp_idx, temp_labels,
        test_size=0.5, stratify=temp_labels, random_state=42
    )
    logger.info(f'Test indices: {len(test_idx)}')
    logger.info(f'Test_idx range: {min(test_idx)} to {max(test_idx)}')
    
    # Full test set inference
    total_samples = 0
    test_idx_counter = 0
    samples_logged = 0
    max_samples_to_log = 10
    
    for batch_idx, (x, y) in enumerate(test_loader):
        try:
            logger.debug(f'Batch {batch_idx}, samples: {x.shape[0]}')
            x_np = x.numpy().astype(np.float32)
            y = y.to(device)
            
            outputs = session.run(None, {'input': x_np})[0]
            outputs = torch.tensor(outputs, device=device)
            
            loss = criterion(outputs, y)
            test_loss += loss.item() * x.shape[0]
            
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(y.cpu().numpy())
            all_inputs.extend(x_np)
            
            batch_size_actual = x.shape[0]
            total_samples += batch_size_actual
            
            # Log first 10 samples
            for i in range(batch_size_actual):
                if samples_logged < max_samples_to_log:
                    input_sample = x_np[i]
                    logit = outputs[i].item()
                    pred = preds[i].item()
                    true = y[i].item()
                    logger.info(
                        f'Sample {samples_logged + 1}: '
                        f'Input shape=({input_length}, {no_feats}), '
                        f'min={input_sample.min():+.5f}, max={input_sample.max():+.5f}, '
                        f'Output: logit={logit:.2f}, pred={int(pred)}, true={int(true)}'
                    )
                    samples_logged += 1
            
            # Metadata
            for i in range(batch_size_actual):
                if test_idx_counter >= len(test_idx):
                    logger.error(f'Error: test_idx_counter={test_idx_counter}, max={len(test_idx)}')
                    break
                global_idx = test_idx[test_idx_counter]
                all_metadata.append(all_rows[sample_idx])
                test_idx_counter += 1
            
        except Exception as e:
            logger.error(f'Error in batch {batch_idx}: {str(e)}')
            raise
    
    logger.info(f'Total processed samples: {total_samples}')
    logger.info(f'Total metadata entries: {len(all_metadata)}')
    test_loss /= len(test_dataset)
    
    # Metrics
    logger.info(f'Predictions: {len(test_preds)}, Labels: {len(test_labels)}')
    if len(test_preds) != len(test_labels):
        logger.error(f'Mismatch: Predictions ({len(test_preds)}) != Labels ({len(test_labels)})')
        raise ValueError('Prediction-label mismatch')
    
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds)
    test_recall = recall_score(test_labels, test_preds)
    
    logger.info(f'Test Loss: {test_loss:.4f}')
    logger.info(f'Test Accuracy: {test_acc:.4f}')
    logger.info(f'Test F1: {test_f1:.4f}')
    logger.info(f'Test Precision: {test_precision:.4f}')
    logger.info(f'Test Recall: {test_recall:.4f}')
    
    # Save all predictions
    output_data = []
    for i in range(len(all_inputs)):
        input_flat = all_inputs[i].reshape(-1)
        input_cols = [f'{col}_t{t+1}' for t in range(input_length) for col in feature_cols]
        row = {
            'track': int(all_metadata[i]['Track']),
            'year': int(all_metadata[i]['Year']),
            'month': int(all_metadata[i]['Month']),
            'day': float(all_metadata[i]['Day']),
            **{col: input_flat[j] for j, col in enumerate(input_cols)},
            'pred_label': int(test_preds[i]),
            'true_label': int(test_labels[i])
        }
        output_data.append(row)
    
    output_df = pd.DataFrame(output_data)
    output_path = os.path.join(data_dir, 'test_predictions.csv')
    output_df.to_csv(output_path, index=False)
    logger.info(f'Saved test predictions to {output_path}')
    
    # Select 5 positive and 5 negative samples
    positive_indices = [i for i, label in enumerate(test_labels) if label == 1]
    negative_indices = [i for i, label in enumerate(test_labels) if label == 0]
    
    if len(positive_indices) < 5:
        logger.warning(f'Only {len(positive_indices)} positive samples available, selecting all.')
    if len(negative_indices) < 5:
        logger.warning(f'Only {len(negative_indices)} negative samples available, selecting all.')
    
    np.random.seed(42)
    selected_positive = np.random.choice(positive_indices, size=min(5, len(positive_indices)), replace=False)
    selected_negative = np.random.choice(negative_indices, size=min(5, len(negative_indices)), replace=False)
    selected_indices = np.concatenate([selected_positive, selected_negative])
    
    # Predict and save selected samples
    sample_data = []
    for i in selected_indices:
        x_np = all_inputs[i].astype(np.float32)
        true_label = test_labels[i]
        
        outputs = session.run(None, {'input': x_np[np.newaxis, ...]})[0][0]
        pred_label = int(torch.sigmoid(torch.tensor(outputs)) >= 0.5)
        
        input_flat = x_np.reshape(-1)
        input_cols = [f'{col}_t{t+1}' for t in range(input_length) for col in feature_cols]
        row = {
            'track': int(all_metadata[i]['Track']),
            'year': int(all_metadata[i]['Year']),
            'month': int(all_metadata[i]['Month']),
            'day': float(all_metadata[i]['Day']),
            **{col: input_flat[j] for j, col in enumerate(input_cols)},
            'pred_label': pred_label,
            'true_label': int(true_label)
        }
        sample_data.append(row)
    
    sample_df = pd.DataFrame(sample_data)
    sample_path = os.path.join(data_dir, 'sample_predictions.csv')
    sample_df.to_csv(sample_path, index=False)
    logger.info(f'Saved {len(sample_data)} selected sample predictions to {sample_path}')
    
    return test_acc, test_f1

if __name__ == '__main__':
    test()