import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import VortexDataset
from model import TSMixerClassifier
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import onnx
import torch.onnx

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler('logs/train.log'),
            logging.StreamHandler()
        ]
    )

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s):
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(train_f1s, label='Train F1')
    plt.plot(val_f1s, label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/training_curves.png')
    plt.close()

def compute_pos_weight(dataset):
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    if pos_count == 0:
        logging.warning('No positive samples in training set. Using default pos_weight=1.0')
        return torch.tensor([1.0])
    pos_weight = neg_count / pos_count
    return torch.tensor([pos_weight])

def export_to_onnx(model, input_length, no_feats, device, onnx_path='models/ts_mixer.onnx'):
    """
    Export the model to ONNX format.
    """
    model.eval()
    dummy_input = torch.randn(1, input_length, no_feats).to(device)  # Batch size 1 for export
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,  # Stable opset
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        # Verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logging.info(f"ONNX model saved and verified at {onnx_path}")
    except Exception as e:
        logging.error(f"Failed to export ONNX model: {str(e)}")
        raise

def train():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    input_length = 8
    no_feats = 284
    feat_mixing_hidden_channels = 512
    no_mixer_layers = 4
    dropout = 0.3
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info('Starting training with hyperparameters:')
    logger.info(f'input_length={input_length}, no_feats={no_feats}, batch_size={batch_size}, epochs={num_epochs}')
    
    data_dir = 'data/data_all_label'
    train_dataset = VortexDataset(data_dir, split='train', input_length=input_length)
    val_dataset = VortexDataset(data_dir, split='val', input_length=input_length)
    test_dataset = VortexDataset(data_dir, split='test', input_length=input_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f'Training samples: {len(train_dataset)}')
    logger.info(f'Validation samples: {len(val_dataset)}')
    logger.info(f'Test samples: {len(test_dataset)}')
    
    pos_weight = compute_pos_weight(train_dataset).to(device)
    logger.info(f'Positive class weight: {pos_weight.item():.4f}')
    
    model = TSMixerClassifier(
        input_length=input_length,
        no_feats=no_feats,
        feat_mixing_hidden_channels=feat_mixing_hidden_channels,
        no_mixer_layers=no_mixer_layers,
        dropout=dropout
    ).to(device)
    logger.info(f'Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}' )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_f1 = 0.0
    os.makedirs('models', exist_ok=True)
    model_path = 'models/ts_mixer.pth'
    onnx_path = 'models/ts_mixer.onnx'
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(y.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)
        
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_path)
            export_to_onnx(model, input_length, no_feats, device, onnx_path)
            logger.info(f'Saved best model with Val F1: {val_f1:.4f}')
        
        logger.info(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}')
        logger.info(f'           Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
    
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s)
    logger.info('Training completed. Curves saved to plots/training_curves.png')

if __name__ == '__main__':
    train()