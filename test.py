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
    
    input_length = 8
    no_feats = 284
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info('Starting testing with hyperparameters:')
    logger.info(f'input_length={input_length}, no_feats={no_feats}, batch_size={batch_size}')
    
    # Data
    data_dir = 'data/data_all_label'
    test_dataset = VortexDataset(data_dir, split='test', input_length=input_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f'Test samples: {len(test_dataset)}')
    
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
    
    # Stratified split indices
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        range(len(all_features)), all_labels,
        test_size=0.3, stratify=all_labels, random_state=42
    )
    val_idx, test_idx, val_labels, test_labels = train_test_split(
        temp_idx, temp_labels,
        test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    # Inference
    for batch_idx, (x, y, _) in enumerate(test_loader):
        x_np = x.numpy().astype(np.float32)  # ONNX expects float32
        y = y.to(device)
        
        # ONNX inference
        outputs = session.run(None, {'input': x_np})[0]  # Shape: (batch_size,)
        outputs = torch.tensor(outputs, device=device)  # Convert to torch for loss
        
        loss = criterion(outputs, y)
        test_loss += loss.item()
        
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(y.cpu().numpy())
        all_inputs.extend(x_np)
        
        # Metadata
        batch_size_actual = x.shape[0]
        for i in range(batch_size_actual):
            sample_idx = batch_idx * batch_size + i
            global_idx = test_idx[sample_idx]
            all_metadata.append(all_rows[global_idx])
    
    test_loss /= len(test_loader)
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds)
    test_recall = recall_score(test_labels, test_preds)
    
    logger.info(f'Test Loss: {test_loss:.4f}')
    logger.info(f'Test Accuracy: {test_acc:.4f}')
    logger.info(f'Test F1 Score: {test_f1:.4f}')
    logger.info(f'Test Precision: {test_precision:.4f}')
    logger.info(f'Test Recall: {test_recall:.4f}')
    
    # Save predictions
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
    
    return test_acc, test_f1

if __name__ == '__main__':
    test()