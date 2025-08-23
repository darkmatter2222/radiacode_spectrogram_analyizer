#!/usr/bin/env python3
"""
Complete Isotope Training Pipeline

This script trains all 21 isotope binary classifiers and generates comprehensive
validation results including confusion matrices, ROC curves, and performance metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
import os
from pathlib import Path
import gc
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                           roc_auc_score, confusion_matrix, roc_curve, 
                           precision_recall_curve, classification_report)
import pandas as pd

class IsotopeCNN(nn.Module):
    """CNN for isotope detection from gamma spectra."""

    def __init__(self, input_size=1024):
        super(IsotopeCNN, self).__init__()

        # 1D Convolutional layers for spectral data
        # Widened channels and slightly larger kernels to capture more detail
        self.conv1 = nn.Conv1d(1, 64, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)

        # Pooling and dropout
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.35)

        # Calculate size after convolutions and pooling
        # After 4 conv+pool layers: 1024 -> 512 -> 256 -> 128 -> 64
        self.fc_input_size = 512 * 64

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Add channel dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Convolutional layers
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))

        return x.squeeze()

class IsotopeDataset(Dataset):
    """Dataset for isotope detection."""
    
    def __init__(self, data_path, isotope_name, max_files=None):
        self.data_path = Path(data_path)
        self.isotope_name = isotope_name
        self.spectra = []
        self.labels = []
        
        print(f"   Loading dataset for {isotope_name}...")
        
        # Load data files - they are stored as JSON batch files in the main directory
        if not self.data_path.exists():
            raise ValueError(f"Data directory not found: {self.data_path}")
        
        # Load ALL batch files (JSON format) - no limit for maximum training data
        batch_files = sorted(list(self.data_path.glob("synthetic_batch_*.json")))
        if max_files:
            batch_files = batch_files[:max_files]
        
        print(f"   Loading ALL {len(batch_files)} batch files for maximum training data...")
        
        for batch_file in batch_files:
            try:
                with open(batch_file, 'r') as f:
                    batch_data = json.load(f)
                
                # Extract spectra and labels for this isotope
                for spectrum_data in batch_data['spectra']:
                    spectrum = spectrum_data['spectrum_counts']
                    labels = spectrum_data['ml_labels']['isotope_presence']
                    
                    self.spectra.append(spectrum)
                    # Get binary label for this isotope (1 if present, 0 if not)
                    binary_label = labels.get(isotope_name, 0)
                    self.labels.append(binary_label)
                
            except Exception as e:
                print(f"   Warning: Could not load {batch_file}: {e}")
                continue  # Continue with other files if one fails
        
        if len(self.spectra) == 0:
            raise ValueError(f"No valid data found for isotope {isotope_name}")
        
        # Convert to tensors
        self.spectra = torch.FloatTensor(np.array(self.spectra))
        self.labels = torch.FloatTensor(np.array(self.labels))

        # Preprocess: sanitize, zero last channel, and normalize per spectrum
        with torch.no_grad():
            # Replace NaN/Inf with zero and clamp negatives
            self.spectra = torch.nan_to_num(self.spectra, nan=0.0, posinf=0.0, neginf=0.0)
            self.spectra = torch.clamp(self.spectra, min=0.0)
            # Always set the last channel to 0
            if self.spectra.dim() == 2 and self.spectra.size(1) > 0:
                self.spectra[:, -1] = 0.0
            # L1 normalize each spectrum (sum to 1); if sum==0, leave zeros
            sums = self.spectra.sum(dim=1, keepdim=True)
            nonzero = sums.squeeze(1) > 0
            self.spectra[nonzero] = self.spectra[nonzero] / sums[nonzero]
        
        
        print(f"   Dataset size: {len(self.spectra):,} samples")
        print(f"   Positive samples: {self.labels.sum().item():,.0f} ({100*self.labels.mean().item():.1f}%)")
        print(f"   Memory usage: ~{self.spectra.nbytes / (1024**3):.2f} GB")
    
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        return self.spectra[idx], self.labels[idx]

def generate_comprehensive_validation(model, test_loader, device, isotope_name, model_dir):
    """Generate comprehensive validation results including confusion matrix, ROC curves, etc."""
    
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    print(f"   Generating comprehensive validation results...")
    
    with torch.no_grad():
        for batch_spectra, batch_labels in test_loader:
            batch_spectra, batch_labels = batch_spectra.to(device), batch_labels.to(device)
            
            outputs = model(batch_spectra)
            probabilities = outputs.cpu().numpy()
            predictions = (outputs > 0.5).float().cpu().numpy()
            labels = batch_labels.cpu().numpy()
            
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)
            all_labels.extend(labels)
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
    roc_auc = roc_auc_score(all_labels, all_probabilities)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(all_labels, all_probabilities)
    
    # Precision-Recall curve
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(all_labels, all_probabilities)
    pr_auc = np.trapz(pr_precision, pr_recall)
    
    # Create validation results directory
    validation_dir = model_dir / f"{isotope_name.replace('-', '_').replace('/', '_')}_validation_results"
    validation_dir.mkdir(exist_ok=True)
    
    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Absent', 'Present'],
                yticklabels=['Absent', 'Present'])
    plt.title(f'Confusion Matrix - {isotope_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(validation_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {isotope_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(validation_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(pr_recall, pr_precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {isotope_name}')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(validation_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Classification report
    class_report = classification_report(all_labels, all_predictions, 
                                       target_names=['Absent', 'Present'],
                                       output_dict=True)
    
    # Save detailed results
    validation_results = {
        'isotope_name': isotope_name,
        'validation_timestamp': datetime.now().isoformat(),
        'performance_metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc)
        },
        'confusion_matrix': {
            'true_negative': int(cm[0, 0]),
            'false_positive': int(cm[0, 1]),
            'false_negative': int(cm[1, 0]),
            'true_positive': int(cm[1, 1])
        },
        'classification_report': class_report,
        'dataset_info': {
            'total_test_samples': len(all_labels),
            'positive_samples': int(np.sum(all_labels)),
            'negative_samples': int(len(all_labels) - np.sum(all_labels)),
            'class_balance': float(np.mean(all_labels))
        }
    }
    
    # Save validation results as JSON
    with open(validation_dir / 'validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    # Save raw predictions for further analysis
    np.savez(validation_dir / 'predictions.npz',
             labels=all_labels,
             predictions=all_predictions,
             probabilities=all_probabilities)
    
    print(f"   Validation complete: {accuracy:.2f}% accuracy, {f1:.3f} F1-score")
    
    return validation_results

def train_single_isotope_complete(isotope_name, data_path, models_dir, epochs=20, batch_size=32, learning_rate=0.001):
    """Train a single isotope classifier with comprehensive validation."""
    
    print(f"\nTRAINING MODEL FOR: {isotope_name}")
    print("="*60)
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Create model directory
    model_name = isotope_name.replace('-', '_').replace('/', '_')
    model_dir = Path(models_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load dataset with ALL available files for maximum training data
        print("Loading dataset...")
        dataset = IsotopeDataset(data_path, isotope_name, max_files=None)  # Use ALL files
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        print(f"   Training: {len(train_dataset):,} samples")
        print(f"   Validation: {len(val_dataset):,} samples")
        print(f"   Test: {len(test_dataset):,} samples")

        # Create data loaders
        pin = (device.type == 'cuda')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

        # Initialize model
        model = IsotopeCNN().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 2), gamma=0.5)
        
        # Training tracking
        training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        # Training
        print(f"\nTraining for {epochs} epochs...")
        print("Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR")
        print("-" * 56)
        
        best_val_accuracy = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_spectra, batch_labels in train_loader:
                batch_spectra, batch_labels = batch_spectra.to(device, non_blocking=True), batch_labels.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(batch_spectra)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
                
                # Clear batch from GPU memory
                del batch_spectra, batch_labels, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_spectra, batch_labels in val_loader:
                    batch_spectra, batch_labels = batch_spectra.to(device, non_blocking=True), batch_labels.to(device, non_blocking=True)
                    
                    outputs = model(batch_spectra)
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
                    
                    # Clear batch from GPU memory
                    del batch_spectra, batch_labels, outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Calculate metrics
            train_accuracy = 100 * train_correct / train_total
            val_accuracy = 100 * val_correct / val_total
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record training history
            training_history['train_loss'].append(train_loss / len(train_loader))
            training_history['train_accuracy'].append(train_accuracy)
            training_history['val_loss'].append(val_loss / len(val_loader))
            training_history['val_accuracy'].append(val_accuracy)
            training_history['learning_rate'].append(current_lr)
            
            print(f"{epoch+1:5d} | {train_loss/len(train_loader):10.4f} | {train_accuracy:9.2f} | "
                  f"{val_loss/len(val_loader):8.4f} | {val_accuracy:7.2f} | {current_lr:.2e}")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = model.state_dict().copy()
            
            scheduler.step()
            
            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Load best model for final validation
        model.load_state_dict(best_model_state)
        
        # Generate comprehensive validation results
        validation_results = generate_comprehensive_validation(
            model, test_loader, device, isotope_name, model_dir
        )
        
        # Save training history plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(training_history['train_loss'], label='Train Loss')
        plt.plot(training_history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(training_history['train_accuracy'], label='Train Accuracy')
        plt.plot(training_history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(training_history['learning_rate'])
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.text(0.1, 0.8, f"Isotope: {isotope_name}", fontsize=12, weight='bold')
        plt.text(0.1, 0.7, f"Best Val Accuracy: {best_val_accuracy:.2f}%", fontsize=11)
        plt.text(0.1, 0.6, f"Test Accuracy: {validation_results['performance_metrics']['accuracy']:.2f}%", fontsize=11)
        plt.text(0.1, 0.5, f"F1 Score: {validation_results['performance_metrics']['f1_score']:.3f}", fontsize=11)
        plt.text(0.1, 0.4, f"ROC AUC: {validation_results['performance_metrics']['roc_auc']:.3f}", fontsize=11)
        plt.text(0.1, 0.3, f"Positive Samples: {validation_results['dataset_info']['positive_samples']:,}", fontsize=11)
        plt.text(0.1, 0.2, f"Training Epochs: {epochs}", fontsize=11)
        plt.axis('off')
        plt.title('Model Summary')
        
        plt.tight_layout()
        plt.savefig(model_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save model and training info
        model_path = model_dir / f"{model_name}_model.pth"
        torch.save(best_model_state, model_path)
        
        # Save training history
        with open(model_dir / 'training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Add training info to validation results
        validation_results['training_info'] = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'best_validation_accuracy': best_val_accuracy,
            'final_train_accuracy': training_history['train_accuracy'][-1],
            'model_path': str(model_path)
        }
        
        # Update validation results file with training info
        validation_dir = model_dir / f"{model_name}_validation_results"
        with open(validation_dir / 'validation_results.json', 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        print(f"\nModel saved: {model_path}")
        print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
        print(f"Test accuracy: {validation_results['performance_metrics']['accuracy']:.2f}%")
        print(f"F1 Score: {validation_results['performance_metrics']['f1_score']:.3f}")
        
        # Final cleanup
        del model, optimizer, scheduler, train_loader, val_loader, test_loader
        del dataset, train_dataset, val_dataset, test_dataset
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'isotope': isotope_name,
            'status': 'completed',
            'best_val_accuracy': best_val_accuracy,
            'test_accuracy': validation_results['performance_metrics']['accuracy'],
            'f1_score': validation_results['performance_metrics']['f1_score'],
            'roc_auc': validation_results['performance_metrics']['roc_auc'],
            'model_path': str(model_path),
            'validation_dir': str(validation_dir)
        }
        
    except Exception as e:
        print(f"Training failed for {isotope_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'isotope': isotope_name,
            'status': 'failed',
            'error': str(e)
        }

def train_all_isotopes_complete():
    """Train all 21 isotope classifiers with comprehensive validation."""
    
    # All 21 isotopes
    isotopes = [
        # Background (4)
        "K-40", "U-238_series_Bi-214", "Th-232_series_Tl-208", "U-235_peak",
        # Calibration (3) 
        "Cs-137", "Ba-133", "Co-57",
        # Industrial (6)
        "Cs-134", "Co-60", "Ir-192", "Se-75", "Yb-169", "Am-241",
        # Medical (8)
        "Tc-99m", "I-131", "I-123", "F-18", "Ga-67", "In-111", "Tl-201", "Xe-133"
    ]
    
    print("COMPREHENSIVE ISOTOPE TRAINING PIPELINE")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training {len(isotopes)} isotope classifiers")
    print("Features: CNN training + confusion matrices + ROC curves + comprehensive validation")
    print()
    
    # Training parameters
    epochs = 3
    batch_size = 32
    learning_rate = 0.001
    data_path = "O:/master_data_collection/isotope"
    models_dir = "models"
    
    print("Training Parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Data path: {data_path}")
    print(f"  Models directory: {models_dir}")
    print()
    
    results = []
    start_time = time.time()
    
    for i, isotope in enumerate(isotopes, 1):
        print(f"[{i}/{len(isotopes)}] Processing {isotope}...")
        
        isotope_start = time.time()
        
        result = train_single_isotope_complete(
            isotope, data_path, models_dir, epochs, batch_size, learning_rate
        )
        
        isotope_time = time.time() - isotope_start
        result['training_time_minutes'] = isotope_time / 60
        
        results.append(result)
        
        if result['status'] == 'completed':
            print(f"[COMPLETED] {isotope} - {result['test_accuracy']:.2f}% accuracy in {isotope_time/60:.1f}m")
        else:
            print(f"[FAILED] {isotope} - {result.get('error', 'Unknown error')}")
        
        print("-" * 60)
    
    total_time = time.time() - start_time
    
    # Generate comprehensive summary
    print("\nFINAL TRAINING SUMMARY")
    print("="*60)
    
    completed = [r for r in results if r['status'] == 'completed']
    failed = [r for r in results if r['status'] != 'completed']
    
    print(f"Completed: {len(completed)}/{len(isotopes)} models")
    print(f"Failed: {len(failed)} models")
    print(f"Total training time: {total_time/3600:.1f} hours")
    
    if completed:
        accuracies = [r['test_accuracy'] for r in completed]
        f1_scores = [r['f1_score'] for r in completed]
        
        print(f"\nPerformance Statistics:")
        print(f"  Average accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
        print(f"  Best accuracy: {max(accuracies):.2f}%")
        print(f"  Worst accuracy: {min(accuracies):.2f}%")
        print(f"  Average F1 score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
        
        print(f"\nCompleted Models (sorted by accuracy):")
        completed_sorted = sorted(completed, key=lambda x: x['test_accuracy'], reverse=True)
        for result in completed_sorted:
            print(f"  {result['isotope']:<20} {result['test_accuracy']:6.2f}% acc, {result['f1_score']:5.3f} F1, {result['training_time_minutes']:4.1f}m")
    
    if failed:
        print(f"\nFailed Models:")
        for result in failed:
            print(f"  {result['isotope']:<20} {result.get('error', 'Unknown error')}")
    
    # Save comprehensive results
    summary = {
        'training_timestamp': datetime.now().isoformat(),
        'training_parameters': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'data_path': data_path
        },
        'total_isotopes': len(isotopes),
        'completed_models': len(completed),
        'failed_models': len(failed),
        'total_training_time_hours': total_time / 3600,
        'results': results
    }
    
    with open(f'{models_dir}/comprehensive_training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {models_dir}/comprehensive_training_summary.json")
    print("All validation results, confusion matrices, and ROC curves saved with each model!")
    
    return results

if __name__ == "__main__":
    # Set matplotlib to non-interactive backend
    plt.switch_backend('Agg')
    
    # Run comprehensive training
    results = train_all_isotopes_complete()
