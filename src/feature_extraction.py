import numpy as np
import librosa
from src.config import *

def extract_mfcc(audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """
    Extract MFCC features from audio.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        
    Returns:
        numpy.ndarray: MFCC features (mean and std)
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Calculate mean and standard deviation
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    
    return np.concatenate([mfccs_mean, mfccs_std])

def extract_mel_spectrogram(audio, sr=SAMPLE_RATE, n_mels=N_MELS):
    """
    Extract Mel spectrogram features.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        n_mels: Number of mel bands
        
    Returns:
        numpy.ndarray: Mel spectrogram features (mean and std)
    """
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    mel_mean = np.mean(mel_spec_db, axis=1)
    mel_std = np.std(mel_spec_db, axis=1)
    
    return np.concatenate([mel_mean, mel_std])

def extract_chroma(audio, sr=SAMPLE_RATE):
    """
    Extract chroma features.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        
    Returns:
        numpy.ndarray: Chroma features (mean and std)
    """
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    
    return np.concatenate([chroma_mean, chroma_std])

def extract_zero_crossing_rate(audio):
    """
    Extract zero crossing rate.
    
    Args:
        audio: Audio time series
        
    Returns:
        numpy.ndarray: ZCR features (mean and std)
    """
    zcr = librosa.feature.zero_crossing_rate(audio)
    
    zcr_flat = zcr.flatten()
    zcr_mean = np.mean(zcr_flat)
    zcr_std = np.std(zcr_flat)
    
    return np.array([zcr_mean, zcr_std])

def extract_spectral_features(audio, sr=SAMPLE_RATE):
    """
    Extract spectral centroid, rolloff, and contrast.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        
    Returns:
        numpy.ndarray: Combined spectral features
    """
    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    centroid_mean = np.mean(spectral_centroid)
    centroid_std = np.std(spectral_centroid)
    
    # Spectral rolloff (85%)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    rolloff_mean = np.mean(spectral_rolloff)
    rolloff_std = np.std(spectral_rolloff)
    
    # Spectral contrast 
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    contrast_mean = np.mean(spectral_contrast, axis=1)
    contrast_std = np.std(spectral_contrast, axis=1)
    
    return np.concatenate([
        [centroid_mean, centroid_std],
        [rolloff_mean, rolloff_std],
        contrast_mean,
        contrast_std
    ])

def extract_rms_energy(audio):
    """
    Extract Root Mean Square (RMS) energy.
        
    Args:
        audio: Audio time series
        
    Returns:
        numpy.ndarray: RMS energy features (mean and std)
    """
    rms = librosa.feature.rms(y=audio)
    
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    
    return np.array([rms_mean, rms_std])

def extract_all_features(audio, sr=SAMPLE_RATE):
    """
    Extract all features from audio signal.
        
    Args:
        audio: Audio time series
        sr: Sample rate
        
    Returns:
        numpy.ndarray: Combined feature vector
    """
    features = []
    
    # Extract each feature type
    features.append(extract_mfcc(audio, sr))
    features.append(extract_mel_spectrogram(audio, sr))
    features.append(extract_chroma(audio, sr))
    features.append(extract_zero_crossing_rate(audio))
    features.append(extract_spectral_features(audio, sr))
    features.append(extract_rms_energy(audio))
    
    return np.concatenate(features)

def get_feature_vector_size():
    """
    Calculate the total size of the feature vector.
    
    Returns:
        int: Size of feature vector
    """
    # MFCC: n_mfcc * 2 (mean + std)
    # Mel: n_mels * 2
    # Chroma: 12 * 2
    # ZCR: 2
    # Spectral: 2 + 2 + 7*2 (centroid, rolloff, contrast)
    # RMS: 2
    
    return (N_MFCC * 2) + (N_MELS * 2) + (12 * 2) + 2 + 18 + 2