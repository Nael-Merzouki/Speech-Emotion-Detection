import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from itertools import product
import json
import os

from src.models import (build_mlp, build_cnn_1d, build_lstm, compile_model, reshape_for_model)
from src.config import MODEL_PATH, RESULTS_PATH

class HyperparameterTuner:
    """
    Hyperparameter tuning using grid search with cross-validation.
    """
    
    def __init__(self, X_train, y_train, model_type='mlp'):
        """
        Initialize tuner.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type: Type of model to tune
        """
        self.X_train = X_train
        self.y_train = y_train
        self.model_type = model_type
        self.feature_dim = X_train.shape[1]
        self.num_classes = len(np.unique(y_train))
        self.results = []
    
    def tune_mlp(self, param_grid, n_splits=3, epochs=50, batch_size=32):
        """
        Tune MLP hyperparameters.
        
        Args:
            param_grid: Dictionary of hyperparameters to tune
            n_splits: Number of CV folds
            epochs: Training epochs
            batch_size: Batch size
            
        Returns:
            pd.DataFrame: Tuning results
        """
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING - MLP")
        print("="*60)
        
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        print(f"Using {n_splits}-fold cross-validation\n")
        
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for idx, params in enumerate(param_combinations, 1):
            print(f"[{idx}/{len(param_combinations)}] Testing: {params}")
            
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(self.X_train), 1):
                X_train_fold = self.X_train[train_idx]
                y_train_fold = self.y_train[train_idx]
                X_val_fold = self.X_train[val_idx]
                y_val_fold = self.y_train[val_idx]
                
                # Build and compile model
                model = build_mlp(
                    self.feature_dim,
                    self.num_classes,
                    hidden_layers=params['hidden_layers'],
                    dropout_rate=params['dropout_rate']
                )
                model = compile_model(model, learning_rate=params['learning_rate'])
                
                # Train
                history = model.fit(
                    X_train_fold, y_train_fold,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_val_fold, y_val_fold),
                    verbose=0
                )
                
                # Get best validation accuracy
                best_val_acc = max(history.history['val_accuracy'])
                fold_scores.append(best_val_acc)
            
            # Calculate average score
            avg_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            result = {
                **params,
                'mean_val_accuracy': avg_score,
                'std_val_accuracy': std_score
            }
            self.results.append(result)
            
            print(f"  Mean Val Accuracy: {avg_score:.4f} (+/- {std_score:.4f})\n")
        
        # Convert to DataFrame and sort
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('mean_val_accuracy', ascending=False)
        
        print("\n" + "="*60)
        print("TUNING RESULTS (Top 5)")
        print("="*60)
        print(results_df.head().to_string(index=False))
        
        return results_df
    
    def tune_cnn(self, param_grid, n_splits=3, epochs=50, batch_size=32):
        """
        Tune CNN hyperparameters.
        
        Args:
            param_grid: Dictionary of hyperparameters to tune
            n_splits: Number of CV folds
            epochs: Training epochs
            batch_size: Batch size
            
        Returns:
            pd.DataFrame: Tuning results
        """
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING - CNN")
        print("="*60)
        
        keys = param_grid.keys()
        values = param_grid.values()
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        print(f"Using {n_splits}-fold cross-validation\n")
        
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Reshape data for CNN
        X_train_reshaped = reshape_for_model(self.X_train, 'cnn')
        
        for idx, params in enumerate(param_combinations, 1):
            print(f"[{idx}/{len(param_combinations)}] Testing: {params}")
            
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_reshaped), 1):
                X_train_fold = X_train_reshaped[train_idx]
                y_train_fold = self.y_train[train_idx]
                X_val_fold = X_train_reshaped[val_idx]
                y_val_fold = self.y_train[val_idx]
                
                # Build and compile model
                model = build_cnn_1d(
                    self.feature_dim,
                    self.num_classes,
                    num_filters=params['num_filters'],
                    dropout_rate=params['dropout_rate']
                )
                model = compile_model(model, learning_rate=params['learning_rate'])
                
                # Train
                history = model.fit(
                    X_train_fold, y_train_fold,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_val_fold, y_val_fold),
                    verbose=0
                )
                
                best_val_acc = max(history.history['val_accuracy'])
                fold_scores.append(best_val_acc)
            
            avg_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            result = {
                **params,
                'mean_val_accuracy': avg_score,
                'std_val_accuracy': std_score
            }
            self.results.append(result)
            
            print(f"  Mean Val Accuracy: {avg_score:.4f} (+/- {std_score:.4f})\n")
        
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('mean_val_accuracy', ascending=False)
        
        print("\n" + "="*60)
        print("TUNING RESULTS (Top 5)")
        print("="*60)
        print(results_df.head().to_string(index=False))
        
        return results_df
    
    def save_results(self, results_df, filename='tuning_results.csv'):
        """
        Save tuning results.
        
        Args:
            results_df: Results DataFrame
            filename: Output filename
        """
        os.makedirs(RESULTS_PATH, exist_ok=True)
        filepath = os.path.join(RESULTS_PATH, filename)
        results_df.to_csv(filepath, index=False)
        print(f"\nTuning results saved to: {filepath}")

def quick_experiment(X_train, y_train, X_test, y_test, 
                     model_builder, model_type, params, epochs=50):
    """
    Quickly test a model configuration.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_builder: Function to build model
        model_type: Type of model
        params: Model parameters
        epochs: Training epochs
        
    Returns:
        dict: Results
    """
    # Reshape if needed
    X_train_shaped = reshape_for_model(X_train, model_type)
    X_test_shaped = reshape_for_model(X_test, model_type)
    
    # Build and train
    model = model_builder(**params)
    model = compile_model(model)
    
    history = model.fit(
        X_train_shaped, y_train,
        batch_size=32,
        epochs=epochs,
        validation_split=0.2,
        verbose=0
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test_shaped, y_test, verbose=0)
    
    return {
        'params': params,
        'train_acc': history.history['accuracy'][-1],
        'val_acc': max(history.history['val_accuracy']),
        'test_acc': test_acc,
        'history': history
    }