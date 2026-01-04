import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import os

from src.models import *
from src.preprocessing import AudioPreprocessor
from src.config import MODEL_PATH, RESULTS_PATH, BATCH_SIZE, EPOCHS

class ModelTrainer:
    """
    Trainer class for emotion recognition models.
    """
    
    def __init__(self, processed_data):
        """
        Initialize trainer with processed data.
        
        Args:
            processed_data: Dictionary containing preprocessed data
        """
        self.X_train = processed_data['X_train']
        self.X_test = processed_data['X_test']
        self.y_train = processed_data['y_train']
        self.y_test = processed_data['y_test']
        self.label_encoder = processed_data['label_encoder']
        self.feature_dim = processed_data['feature_dim']
        self.num_classes = len(self.label_encoder.classes_)
        self.model = None
        self.history = None
        
    def build_and_compile_model(self, model_type='mlp', **kwargs):
        """
        Build and compile a model.
        
        Args:
            model_type: Type of model to build
            **kwargs: Additional arguments for model building
            
        Returns:
            keras.Model: Compiled model
        """
        print(f"\nBuilding {model_type.upper()} model...")
        
        if model_type == 'mlp':
            self.model = build_mlp(self.feature_dim, self.num_classes, **kwargs)
        elif model_type == 'cnn':
            self.model = build_cnn_1d(self.feature_dim, self.num_classes, **kwargs)
        elif model_type == 'lstm':
            self.model = build_lstm(self.feature_dim, self.num_classes, **kwargs)
        elif model_type == 'cnn_lstm':
            self.model = build_cnn_lstm(self.feature_dim, self.num_classes, **kwargs)
        elif model_type == 'attention':
            self.model = build_attention_model(self.feature_dim, self.num_classes, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = compile_model(self.model)
        print_model_summary(self.model)
        
        return self.model
    
    def train(self, model_type='mlp', batch_size=BATCH_SIZE, epochs=EPOCHS, 
              validation_split=0.2, **model_kwargs):
        """
        Train the model.
        
        Args:
            model_type: Type of model to train
            batch_size: Training batch size
            epochs: Maximum number of epochs
            validation_split: Validation split ratio
            **model_kwargs: Additional model building arguments
            
        Returns:
            History: Training history
        """
        # Build model
        self.build_and_compile_model(model_type, **model_kwargs)
        
        # Reshape data for model
        X_train_reshaped = reshape_for_model(self.X_train, model_type)
        X_test_reshaped = reshape_for_model(self.X_test, model_type)
        
        # Get callbacks
        callbacks = get_callbacks(model_type)
        
        # Train model
        print(f"\nTraining {model_type.upper()} model...")
        print(f"Training samples: {len(X_train_reshaped)}")
        print(f"Validation split: {validation_split}")
        print(f"Batch size: {batch_size}")
        print(f"Max epochs: {epochs}")
        print("=" * 60)
        
        self.history = self.model.fit(
            X_train_reshaped,
            self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print("\n" + "=" * 60)
        print("EVALUATING ON TEST SET")
        print("=" * 60)
        
        test_loss, test_accuracy = self.model.evaluate(
            X_test_reshaped,
            self.y_test,
            verbose=0
        )
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return self.history
    
    def evaluate(self, model_type='mlp'):
        """
        Evaluate model and generate predictions.
        
        Args:
            model_type: Type of model for reshaping
            
        Returns:
            tuple: (predictions, true_labels)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        X_test_reshaped = reshape_for_model(self.X_test, model_type)
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test_reshaped, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        return y_pred, self.y_test
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history (loss and accuracy).
        
        Args:
            save_path: Path to save plot
        """
        if self.history is None:
            raise ValueError("No training history available!")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axes[0].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[1].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_pred, y_true, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            y_pred: Predicted labels
            y_true: True labels
            save_path: Path to save plot
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Get class names
        class_names = self.label_encoder.classes_
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
        
        # Print normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        print("\nNormalized Confusion Matrix:")
        print("=" * 60)
        cm_df = pd.DataFrame(
            cm_normalized,
            index=class_names,
            columns=class_names
        )
        print(cm_df.round(3))
    
    def print_classification_report(self, y_pred, y_true):
        """
        Print detailed classification report.
        
        Args:
            y_pred: Predicted labels
            y_true: True labels
        """
        class_names = self.label_encoder.classes_
        
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4
        ))
    
    def save_results(self, model_type, y_pred, y_true):
        """
        Save training results and predictions.
        
        Args:
            model_type: Type of model
            y_pred: Predicted labels
            y_true: True labels
        """
        os.makedirs(RESULTS_PATH, exist_ok=True)
        
        # Save history
        history_file = os.path.join(RESULTS_PATH, f'{model_type}_history.pkl')
        with open(history_file, 'wb') as f:
            pickle.dump(self.history.history, f)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'true_label': self.label_encoder.inverse_transform(y_true),
            'predicted_label': self.label_encoder.inverse_transform(y_pred),
            'correct': y_true == y_pred
        })
        
        predictions_file = os.path.join(RESULTS_PATH, f'{model_type}_predictions.csv')
        predictions_df.to_csv(predictions_file, index=False)
        
        print(f"\nResults saved:")
        print(f"  - History: {history_file}")
        print(f"  - Predictions: {predictions_file}")

def compare_models(results_dict):
    """
    Compare multiple models' performance.
    
    Args:
        results_dict: Dictionary with model names as keys and 
                     (accuracy, history) tuples as values
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    comparison_df = pd.DataFrame({
        'Model': list(results_dict.keys()),
        'Test Accuracy': [acc for acc, _ in results_dict.values()],
        'Best Val Accuracy': [max(hist.history['val_accuracy']) 
                             for _, hist in results_dict.values()],
        'Final Training Accuracy': [hist.history['accuracy'][-1] 
                                   for _, hist in results_dict.values()]
    })
    
    comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)
    print(comparison_df.to_string(index=False))
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comparison_df))
    width = 0.35
    
    ax.bar(x - width/2, comparison_df['Test Accuracy'], width, 
           label='Test Accuracy', alpha=0.8)
    ax.bar(x + width/2, comparison_df['Best Val Accuracy'], width, 
           label='Best Val Accuracy', alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Performance Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Model'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'plots', 'model_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Example usage
    from preprocessing import AudioPreprocessor
    
    print("Loading processed data...")
    preprocessor = AudioPreprocessor()
    processed_data = preprocessor.load_processed_data('ravdess_processed.pkl')
    
    print("\nInitializing trainer...")
    trainer = ModelTrainer(processed_data)
    
    print("\nTraining MLP model...")
    history = trainer.train(model_type='mlp', epochs=50)
    
    print("\nEvaluating model...")
    y_pred, y_true = trainer.evaluate(model_type='mlp')
    
    trainer.plot_training_history()
    trainer.plot_confusion_matrix(y_pred, y_true)
    trainer.print_classification_report(y_pred, y_true)