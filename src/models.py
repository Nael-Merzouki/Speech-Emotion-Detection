import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
from src.config import MODEL_PATH, LEARNING_RATE, BATCH_SIZE, EPOCHS

def build_mlp(input_dim, num_classes, hidden_layers=[256, 128, 64], dropout_rate=0.3):
    """
    Build a Multi-Layer Perceptron (MLP) model.
    
    Simple feedforward neural network with dense layers.
    Good baseline model for feature-based classification.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        hidden_layers: List of hidden layer sizes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        keras.Model: Compiled MLP model
    """
    model = models.Sequential(name='MLP')
    
    # Input layer
    model.add(layers.Input(shape=(input_dim,)))
    
    # Hidden layers
    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(
            units, 
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            name=f'dense_{i+1}'
        ))
        model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
        model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))
    
    return model

def build_cnn_1d(input_dim, num_classes, num_filters=[64, 128, 256], dropout_rate=0.3):
    """
    Build a 1D Convolutional Neural Network.
    
    Uses 1D convolutions to learn patterns in feature sequences.
    Treats features as a 1D sequence.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        num_filters: List of filter sizes for conv layers
        dropout_rate: Dropout rate for regularization
        
    Returns:
        keras.Model: Compiled 1D CNN model
    """
    model = models.Sequential(name='CNN_1D')
    
    # Reshape input to add channel dimension (samples, features, channels)
    model.add(layers.Input(shape=(input_dim, 1)))
    
    # Convolutional blocks
    for i, filters in enumerate(num_filters):
        model.add(layers.Conv1D(
            filters, 
            kernel_size=3, 
            activation='relu',
            padding='same',
            name=f'conv1d_{i+1}'
        ))
        model.add(layers.BatchNormalization(name=f'bn_conv_{i+1}'))
        model.add(layers.MaxPooling1D(pool_size=2, name=f'maxpool_{i+1}'))
        model.add(layers.Dropout(dropout_rate, name=f'dropout_conv_{i+1}'))
    
    # Flatten and dense layers
    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dense(128, activation='relu', name='dense_1'))
    model.add(layers.BatchNormalization(name='bn_dense'))
    model.add(layers.Dropout(dropout_rate, name='dropout_dense'))
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))
    
    return model

def build_lstm(input_dim, num_classes, lstm_units=[128, 64], dropout_rate=0.3):
    """
    Build a Long Short-Term Memory (LSTM) model.
    
    LSTM networks are good at learning temporal patterns.
    Useful for sequential audio features.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        lstm_units: List of LSTM layer sizes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        keras.Model: Compiled LSTM model
    """
    model = models.Sequential(name='LSTM')
    
    # Reshape input to add time dimension (samples, timesteps, features)
    model.add(layers.Input(shape=(input_dim, 1)))
    
    # LSTM layers
    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1
        model.add(layers.LSTM(
            units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate * 0.5,
            name=f'lstm_{i+1}'
        ))
        model.add(layers.BatchNormalization(name=f'bn_lstm_{i+1}'))
    
    # Dense layers
    model.add(layers.Dense(64, activation='relu', name='dense_1'))
    model.add(layers.Dropout(dropout_rate, name='dropout_dense'))
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))
    
    return model

def build_cnn_lstm(input_dim, num_classes, cnn_filters=[64, 128], 
                   lstm_units=[64], dropout_rate=0.3):
    """
    Build a hybrid CNN-LSTM model.
    
    Combines CNN for feature extraction with LSTM for temporal modeling.
    Often performs better than either architecture alone.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        cnn_filters: List of CNN filter sizes
        lstm_units: List of LSTM layer sizes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        keras.Model: Compiled CNN-LSTM model
    """
    model = models.Sequential(name='CNN_LSTM')
    
    # Input with time and feature dimensions
    model.add(layers.Input(shape=(input_dim, 1)))
    
    # CNN layers for feature extraction
    for i, filters in enumerate(cnn_filters):
        model.add(layers.Conv1D(
            filters,
            kernel_size=3,
            activation='relu',
            padding='same',
            name=f'conv1d_{i+1}'
        ))
        model.add(layers.BatchNormalization(name=f'bn_conv_{i+1}'))
        model.add(layers.MaxPooling1D(pool_size=2, name=f'maxpool_{i+1}'))
        model.add(layers.Dropout(dropout_rate * 0.5, name=f'dropout_conv_{i+1}'))
    
    # LSTM layers for temporal modeling
    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1
        model.add(layers.LSTM(
            units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            name=f'lstm_{i+1}'
        ))
        model.add(layers.BatchNormalization(name=f'bn_lstm_{i+1}'))
    
    # Output layers
    model.add(layers.Dense(64, activation='relu', name='dense_1'))
    model.add(layers.Dropout(dropout_rate, name='dropout_dense'))
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))
    
    return model

def build_attention_model(input_dim, num_classes, lstm_units=128, dropout_rate=0.3):
    """
    Build an LSTM model with attention mechanism.
    
    Attention allows the model to focus on the most important features.
    Can improve performance on complex patterns.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate for regularization
        
    Returns:
        keras.Model: Compiled attention model
    """
    # Input layer
    inputs = layers.Input(shape=(input_dim, 1), name='input')
    
    # LSTM layer with return sequences for attention
    lstm_out = layers.LSTM(
        lstm_units,
        return_sequences=True,
        dropout=dropout_rate,
        name='lstm'
    )(inputs)
    
    # Attention mechanism
    attention = layers.Dense(1, activation='tanh', name='attention_dense')(lstm_out)
    attention = layers.Flatten(name='attention_flatten')(attention)
    attention = layers.Activation('softmax', name='attention_softmax')(attention)
    attention = layers.RepeatVector(lstm_units, name='attention_repeat')(attention)
    attention = layers.Permute([2, 1], name='attention_permute')(attention)
    
    # Apply attention weights
    attended = layers.multiply([lstm_out, attention], name='attention_multiply')
    attended = layers.Lambda(lambda x: keras.backend.sum(x, axis=1), name='attention_sum')(attended)
    
    # Output layers
    dense = layers.Dense(64, activation='relu', name='dense_1')(attended)
    dense = layers.BatchNormalization(name='bn')(dense)
    dense = layers.Dropout(dropout_rate, name='dropout')(dense)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(dense)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='Attention_LSTM')
    
    return model

def compile_model(model, learning_rate=LEARNING_RATE):
    """
    Compile model with optimizer, loss, and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        
    Returns:
        keras.Model: Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_callbacks(model_name, patience=15):
    """
    Get training callbacks for model optimization.
    
    Args:
        model_name: Name for saving model checkpoints
        patience: Patience for early stopping
        
    Returns:
        list: List of callbacks
    """
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Save best model
        ModelCheckpoint(
            filepath=os.path.join(MODEL_PATH, f'{model_name}_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks

def reshape_for_model(X, model_type):
    """
    Reshape input data based on model type.
    
    Args:
        X: Input features
        model_type: Type of model ('mlp', 'cnn', 'lstm', 'cnn_lstm', 'attention')
        
    Returns:
        numpy.ndarray: Reshaped input
    """
    if model_type == 'mlp':
        # MLP expects 2D input (samples, features)
        return X
    elif model_type in ['cnn', 'lstm', 'cnn_lstm', 'attention']:
        # CNN/LSTM expect 3D input (samples, features, channels)
        return X.reshape(X.shape[0], X.shape[1], 1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def print_model_summary(model):
    """
    Print detailed model summary.
    
    Args:
        model: Keras model
    """
    print("\n" + "=" * 60)
    print(f"MODEL: {model.name}")
    print("=" * 60)
    model.summary()
    print("=" * 60)
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    # Test model building
    input_dim = 338  # Example feature dimension
    num_classes = 8
    
    print("Testing model architectures...\n")
    
    models_to_test = {
        'MLP': build_mlp(input_dim, num_classes),
        'CNN_1D': build_cnn_1d(input_dim, num_classes),
        'LSTM': build_lstm(input_dim, num_classes),
        'CNN_LSTM': build_cnn_lstm(input_dim, num_classes),
        'Attention': build_attention_model(input_dim, num_classes)
    }
    
    for name, model in models_to_test.items():
        model = compile_model(model)
        print_model_summary(model)