import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "RAVDESS")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "saved_models")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results")

# Audio parameters
SAMPLE_RATE = 22050  # Standard sample rate for librosa
DURATION = 3  # Maximum duration in seconds
OFFSET = 0.5  # Start reading after this offset

# Feature extraction parameters
N_MFCC = 40  # Number of MFCC coefficients
N_MELS = 128  # Number of mel bands
HOP_LENGTH = 512  # Number of samples between successive frames
N_FFT = 2048  # FFT window size

# Emotion labels
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprised"
}

# Model parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
RANDOM_STATE = 42