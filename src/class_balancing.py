import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

def compute_class_weights(y_train):
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        y_train: Training labels
        
    Returns:
        dict: Class weights
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    
    class_weights = dict(zip(classes, weights))
    
    print("\nClass Weights:")
    print("="*40)
    for cls, weight in class_weights.items():
        print(f"Class {cls}: {weight:.4f}")
    
    return class_weights

def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE for oversampling minority classes.
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    print("\nApplying SMOTE...")
    print("="*40)
    print(f"Original distribution: {Counter(y_train)}")
    
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE: {Counter(y_resampled)}")
    print(f"Original samples: {len(X_train)}")
    print(f"After SMOTE: {len(X_resampled)}")
    
    return X_resampled, y_resampled

def apply_adasyn(X_train, y_train, random_state=42):
    """
    Apply ADASYN for adaptive synthetic sampling.
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    print("\nApplying ADASYN...")
    print("="*40)
    print(f"Original distribution: {Counter(y_train)}")
    
    adasyn = ADASYN(random_state=random_state)
    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
    
    print(f"After ADASYN: {Counter(y_resampled)}")
    print(f"Original samples: {len(X_train)}")
    print(f"After ADASYN: {len(X_resampled)}")
    
    return X_resampled, y_resampled

def apply_undersampling(X_train, y_train, random_state=42):
    """
    Apply random undersampling.
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    print("\nApplying Random Undersampling...")
    print("="*40)
    print(f"Original distribution: {Counter(y_train)}")
    
    rus = RandomUnderSampler(random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    
    print(f"After undersampling: {Counter(y_resampled)}")
    print(f"Original samples: {len(X_train)}")
    print(f"After undersampling: {len(X_resampled)}")
    
    return X_resampled, y_resampled

if __name__ == "__main__":
    # Test balancing
    from src.preprocessing import AudioPreprocessor
    
    preprocessor = AudioPreprocessor()
    processed_data = preprocessor.load_processed_data('ravdess_processed.pkl')
    
    X_train = processed_data['X_train']
    y_train = processed_data['y_train']
    
    # Compute class weights
    class_weights = compute_class_weights(y_train)
    
    # Test SMOTE
    X_smote, y_smote = apply_smote(X_train, y_train)