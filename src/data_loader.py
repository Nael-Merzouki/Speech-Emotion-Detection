import os
import librosa
import numpy as np
import pandas as pd
from src.config import DATA_PATH, EMOTIONS, SAMPLE_RATE, DURATION, OFFSET

def load_ravdess_data():
    """
    Load RAVDESS dataset and extract metadata from filenames.
    
    Returns:
        pd.DataFrame: DataFrame with file paths and metadata
    """
    data = []
    
    # Walk through all actor folders
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                
                # Get file metadata from filename
                parts = file.split("-")

                modality = parts[0]
                vocal_channel = parts[1]
                emotion_code = parts[2]
                emotion = EMOTIONS.get(emotion_code, "unknown")
                intensity = "normal" if parts[3] == "01" else "strong"
                statement = parts[4]
                repetition = parts[5]
                actor = int(parts[6][:2])
                gender = "male" if actor % 2 == 1 else "female"

                data.append({
                        "file_path": file_path,
                        "filename": file,
                        "emotion": emotion,
                        "emotion_code": emotion_code,
                        "intensity": intensity,
                        "statement": statement,
                        "repetition": repetition,
                        "actor": actor,
                        "gender": gender
                    })
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} audio files")
    print(f"\nEmotion distribution:\n{df["emotion"].value_counts()}")
    
    return df

def load_audio_file(file_path, sr=SAMPLE_RATE, duration=DURATION, offset=OFFSET):
    """
    Load an audio file and return the waveform.
    
    Args:
        file_path: Path to audio file
        sr: Sample rate
        duration: Maximum duration to load
        offset: Start reading after this offset
        
    Returns:
        numpy.ndarray: Audio time series
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr, duration=duration, offset=offset)
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_audio_info(file_path):
    """
    Get basic information about an audio file.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        dict: Audio metadata
    """
    audio, sr = librosa.load(file_path, sr=None)
    
    return {
        "duration": librosa.get_duration(y=audio, sr=sr),
        "sample_rate": sr,
        "samples": len(audio),
        "channels": 1 if audio.ndim == 1 else audio.shape[0]
    }