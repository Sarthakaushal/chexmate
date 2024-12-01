import torch
import torch.nn as nn
import torch.optim as optim
from models.densenet import DenseNetBinaryClassifier
from dataloader.global_dataset import get_data_loaders
from tqdm import tqdm
import logging
import os
from datetime import datetime
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import argparse

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_recall = None
        self.early_stop = False

    def __call__(self, val_recall):
        if self.best_recall is None:
            self.best_recall = val_recall
        elif val_recall <= self.best_recall + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_recall = val_recall
            self.counter = 0

def setup_logging(log_dir="logs"):
    """Setup logging configuration"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
            
        images, labels = images.to(device), labels.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predicted = (outputs.squeeze() > 0.5).float()
        
        # Store predictions and labels for metrics calculation
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        
        # Update progress bar with current loss
        pbar.set_postfix({'loss': loss.item()})
    
    # Calculate metrics
    epoch_loss = running_loss / len(train_loader)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_predictions)
    
    return epoch_loss, precision, recall, f1, auc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            # Convert single channel images to 3 channels if necessary
            if images.size(1) == 1:
                images = images.repeat(1, 3, 1, 1)
                
            images, labels = images.to(device), labels.to(device).float()
            
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            
            running_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            
            # Store predictions and labels for metrics calculation
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Calculate metrics
    val_loss = running_loss / len(val_loader)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_predictions)
    
    return val_loss, precision, recall, f1, auc

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            # Convert single channel images to 3 channels if necessary
            if images.size(1) == 1:
                images = images.repeat(1, 3, 1, 1)
                
            images = images.to(device)
            outputs = model(images)
            predicted = (outputs.squeeze() > 0.5).float()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    return classification_report(
        all_labels, 
        all_predictions, 
        target_names=['Normal', 'TB'],
        digits=4
    )

def main():
    parser = argparse.ArgumentParser(description='Train DenseNet model for TB classification')
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained',
                      help='Do not use pretrained weights for backbone')
    parser.add_argument('--no-freeze', action='store_false', dest='freeze_backbone',
                      help='Do not freeze backbone weights during training')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=5,
                      help='Early stopping patience (default: 5)')
    parser.add_argument('--load-model', type=str,
                      help='Path to pretrained model checkpoint to load')
    parser.add_argument('--eval-only', action='store_true',
                      help='Only run evaluation on the test set')
    parser.add_argument('--test-dataset', type=str, default='indian',
                      choices=['indian', 'global', 'both'],
                      help='Dataset to evaluate on (default: indian)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Hyperparameters from args
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    patience = args.patience
    
    # Dataset paths
    tb_data_path = "data/TB_Chest_Radiography_Database"
    indian_data_path = "data/indian_dataset"
    
    # Get data loaders
    loaders = get_data_loaders(tb_data_path, indian_data_path, batch_size=batch_size)
    train_loader = loaders['tb_train']
    val_loader = loaders['tb_val']
    indian_loader = loaders['indian_whole']
    global_loader = loaders['global_whole']
    
    # Initialize model with args
    model = DenseNetBinaryClassifier(
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone
    )
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Load pretrained model if specified
    start_epoch = 0
    best_recall = 0.0
    if args.load_model:
        if os.path.isfile(args.load_model):
            logging.info(f"Loading checkpoint '{args.load_model}'")
            checkpoint = torch.load(args.load_model, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if not args.eval_only:  # Only load optimizer if we're training
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_recall = checkpoint.get('val_recall', 0.0)
            logging.info(f"Loaded checkpoint (epoch {checkpoint['epoch']})")
        else:
            logging.error(f"No checkpoint found at '{args.load_model}'")
            return

    # Log configuration
    logging.info(f"Model configuration:")
    logging.info(f"- Pretrained weights: {args.pretrained}")
    logging.info(f"- Backbone frozen: {args.freeze_backbone}")
    logging.info(f"- Batch size: {batch_size}")
    logging.info(f"- Learning rate: {learning_rate}")
    logging.info(f"- Max epochs: {num_epochs}")
    logging.info(f"- Early stopping patience: {patience}")
    if args.load_model:
        logging.info(f"- Loaded model from: {args.load_model}")
    
    # If eval only, run evaluation and exit
    if args.eval_only:
        logging.info("\nEvaluating on Indian dataset:")
        if args.test_dataset == "indian":
            indian_report = evaluate_model(model, indian_loader, device)
            logging.info(f"\nClassification Report on Indian Dataset:\n{indian_report}")
        elif args.test_dataset == "global":
            global_report = evaluate_model(model, global_loader, device)
            logging.info(f"\nClassification Report on Global Dataset:\n{global_report}")
        elif args.test_dataset == "both":
            indian_report = evaluate_model(model, indian_loader, device)
            global_report = evaluate_model(model, global_loader, device)
            logging.info(f"\nClassification Report on Indian Dataset:\n{indian_report}")
            logging.info(f"\nClassification Report on Global Dataset:\n{global_report}")
        else:
            logging.error("Invalid test dataset specified. Use 'global' or 'indian'.")
        return

    # Training loop
    criterion = nn.BCELoss()
    early_stopping = EarlyStopping(patience=patience)
    
    for epoch in range(start_epoch, num_epochs):
        logging.info(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_prec, train_rec, train_f1, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device)
        logging.info(
            f"Training - Loss: {train_loss:.4f}, Recall: {train_rec:.4f}, "
            f"Precision: {train_prec:.4f}, F1: {train_f1:.4f}, AUC-ROC: {train_auc:.4f}"
        )
        
        # Validate
        val_loss, val_prec, val_rec, val_f1, val_auc = validate(
            model, val_loader, criterion, device)
        logging.info(
            f"Validation - Loss: {val_loss:.4f}, Recall: {val_rec:.4f}, "
            f"Precision: {val_prec:.4f}, F1: {val_f1:.4f}, AUC-ROC: {val_auc:.4f}"
        )
        
        # Save best model based on recall
        if val_rec > best_recall:
            best_recall = val_rec
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_recall': val_rec,
            }, 'outputs/saved_models/best_model.pth')
            logging.info(f"Saved new best model with validation recall: {val_rec:.4f}")
        
        # Early stopping
        early_stopping(val_rec)
        if early_stopping.early_stop:
            logging.info(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    # Load best model for evaluation
    logging.info("\nLoading best model for Indian dataset evaluation...")
    checkpoint = torch.load('outputs/saved_models/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on Indian dataset
    logging.info("\nEvaluating on Indian dataset:")
    indian_report = evaluate_model(model, indian_loader, device)
    logging.info(f"\nClassification Report on Indian Dataset:\n{indian_report}")

if __name__ == "__main__":
    
    main()
    
# # Train from scratch
# python baseline.py

# # Resume training from checkpoint
# python baseline.py --load-model outputs/saved_models/checkpoint.pth

# # Only evaluate a trained model
# python baseline.py --load-model outputs/saved_models/best_model.pth --eval-only --test-dataset global

# # Resume training with different learning rate
# python baseline.py --load-model outputs/saved_models/checkpoint.pth --lr 0.0001