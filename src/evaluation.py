import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, precision_recall_fscpre_support, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import os

from src.config import RESULTS_PATH

class ModelEvaluator:
    """
    Model evaluation tools.
    """
    def __init__(self, label_encoder):
        self.label_encoder = label_encoder
        self.class_names = label_encoder.classes_
        self.num_classes = len(self.class_names)
    
    def evaluate_model(self,y_true, y_pred, y_pred_proba=None, model_name="Model"):
        print("\n" + "="*60)
        print(f"EVALUATION: {model_name}")
        print("="*60)
        
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Per-class metrics
        print(f"\nPer-Class Metrics:")
        print("-" * 60)
        
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(y_true, y_pred, average=None)
        
        metrics_df = pd.DataFrame({
            'Emotion': self.class_names,
            'Precision': precision_per_class,
            'Recall': recall_per_class,
            'F1-Score': f1_per_class,
            'Support': support_per_class
        })
        
        print(metrics_df.to_string(index=False))
        
        # Store metrics
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_class_metrics': metrics_df,
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        return results
    
    def plot_per_class_metrics(self, metrics_df, save_path=None):
        """
        Plot per-class precision, recall, and F1-score.
        
        Args:
            metrics_df: DataFrame with per-class metrics
            save_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metrics_df))
        width = 0.25
        
        ax.bar(x - width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
        ax.bar(x, metrics_df['Recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Emotion', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df['Emotion'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix_advanced(self, y_true, y_pred, normalize=False, 
                                       save_path=None):
        """
        Plot advanced confusion matrix with percentages.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize by row
            save_path: Path to save plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='YlOrRd',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Percentage' if normalize else 'Count'},
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_errors(self, y_true, y_pred):
        """
        Analyze prediction errors in detail.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            pd.DataFrame: Error analysis
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Find most confused pairs
        confused_pairs = []
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append({
                        'True Emotion': self.class_names[i],
                        'Predicted Emotion': self.class_names[j],
                        'Count': cm[i, j],
                        'Percentage': cm[i, j] / cm[i].sum() * 100
                    })
        
        errors_df = pd.DataFrame(confused_pairs)
        errors_df = errors_df.sort_values('Count', ascending=False)
        
        print("\n" + "="*60)
        print("ERROR ANALYSIS - Most Confused Pairs")
        print("="*60)
        print(errors_df.head(10).to_string(index=False))
        
        return errors_df
    
    def plot_roc_curves(self, y_true, y_pred_proba, save_path=None):
        """
        Plot ROC curves for multi-class classification.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            save_path: Path to save plot
        """
        # Binarize labels for multi-class ROC
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))
        
        for i, color in zip(range(self.num_classes), colors):
            ax.plot(
                fpr[i], tpr[i], color=color, lw=2,
                label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.2f})'
            )
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title('ROC Curves - Multi-class Classification', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print average AUC
        avg_auc = np.mean(list(roc_auc.values()))
        print(f"\nAverage AUC: {avg_auc:.4f}")
    
    def plot_prediction_confidence(self, y_true, y_pred_proba, save_path=None):
        """
        Plot prediction confidence distribution.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            save_path: Path to save plot
        """
        # Get max probability for each prediction
        max_probs = np.max(y_pred_proba, axis=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        correct = y_true == y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Distribution of confidence scores
        axes[0].hist(max_probs[correct], bins=30, alpha=0.7, 
                    label='Correct', color='green', edgecolor='black')
        axes[0].hist(max_probs[~correct], bins=30, alpha=0.7, 
                    label='Incorrect', color='red', edgecolor='black')
        axes[0].set_xlabel('Prediction Confidence', fontweight='bold')
        axes[0].set_ylabel('Frequency', fontweight='bold')
        axes[0].set_title('Prediction Confidence Distribution', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Confidence vs accuracy
        confidence_bins = np.linspace(0, 1, 11)
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        bin_accuracies = []
        bin_counts = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i+1])
            if mask.sum() > 0:
                bin_accuracies.append(correct[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        axes[1].bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, 
                   color='skyblue', edgecolor='black')
        axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        axes[1].set_xlabel('Prediction Confidence', fontweight='bold')
        axes[1].set_ylabel('Accuracy', fontweight='bold')
        axes[1].set_title('Confidence vs Accuracy', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print statistics
        print("\nConfidence Statistics:")
        print(f"  Correct predictions - Mean confidence: {max_probs[correct].mean():.4f}")
        print(f"  Incorrect predictions - Mean confidence: {max_probs[~correct].mean():.4f}")