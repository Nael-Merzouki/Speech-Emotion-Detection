import numpy as np
import librosa

from src.config import *

def add_noise(audio, noise_factor=0.005):
    """
    Add random noise to audio signal.
    
    Args:
        audio: Audio time series
        noise_factor: Amount of noise to add
        
    Returns:
        numpy.ndarray: Augmented audio
    """
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise

    # Normalize to prevent clipping
    augmented_audio = augmented_audio.astype(type(audio[0]))
    
    return augmented_audio

def shift_time(audio, shift_max=0.2, sr=SAMPLE_RATE):
    """
    Shift audio in time (left or right).
    
    Args:
        audio: Audio time series
        shift_max: Maximum shift as fraction of audio length
        sr: Sample rate
        
    Returns:
        numpy.ndarray: Time-shifted audio
    """
    shift = np.random.randint(int(sr * shift_max))
    direction = np.random.choice(["left", "right"])

    if direction == "right":
        shifted_audio = np.roll(audio, shift)
        shifted_audio[:shift] = 0
    else:
        shifted_audio = np.roll(audio, -shift)
        shifted_audio[-shift:] = 0
    
    return shifted_audio

def change_pitch(audio, sr=SAMPLE_RATE, pitch_factor=None):
    """
    Change the pitch of audio without changing speed.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        pitch_factor: Semitones to shift (if None, random between -3 and 3)
        
    Returns:
        numpy.ndarray: Pitch-shifted audio
    """
    if pitch_factor is None:
        pitch_factor = np.random.uniform(-3, 3)
    
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_factor)

def change_speed(audio, speed_factor=None):
    """
    Change the speed of audio.
    
    Args:
        audio: Audio time series
        speed_factor: Speed multiplier (if None, random between 0.8 and 1.2)
        
    Returns:
        numpy.ndarray: Speed-changed audio
    """
    if speed_factor is None:
        speed_factor = np.random.uniform(0.8, 1.2)
    
    return librosa.effects.time_stretch(audio, rate=speed_factor)

def add_reverb(audio, sr=SAMPLE_RATE, room_scale=0.5):
    """
    Add simple reverb effect to audio.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        room_scale: Room size factor (0 to 1)
        
    Returns:
        numpy.ndarray: Audio with reverb
    """
    delay_samples = int(sr * room_scale * 0.1)
    reverb = np.zeros(len(audio) + delay_samples)

    # Original signal
    reverb[:len(audio)] += audio

    # Add delayed echoes with decreasing amplitude
    for i in range(1, 4):
        delay = delay_samples * i // 3
        amplitude = 0.3 / i
        if delay < len(audio):
            reverb[delay:delay + len(audio)] += audio * amplitude
    
    # Normalize
    reverb = reverb[:len(audio)]
    reverb = reverb / np.max(np.abs(reverb))

    return reverb.astype(audio.dtype)

def augment_audio(audio, sr=SAMPLE_RATE, augmentation_type="random"):
    """
    Apply augmentation to audio signal.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        augmentation_type: Type of augmentation 
            ('noise', 'shift', 'pitch', 'speed', 'reverb', 'random', 'all')
        
    Returns:
        numpy.ndarray: Augmented audio
    """
    if augmentation_type == "random":
        augmentation_type = np.random.choice(["noise", "shift", "pitch", "speed", "reverb"])
    
    if augmentation_type == "noise":
        return add_noise(audio)
    elif augmentation_type == "shift":
        return shift_time(audio, sr=sr)
    elif augmentation_type == "pitch":
        return change_pitch(audio, sr=sr)
    elif augmentation_type == "speed":
        return change_speed(audio)
    elif augmentation_type == "reverb":
        return add_reverb(audio, sr=sr)
    elif augmentation_type == "all":
        # Apply multiple augmentations
        audio = add_noise(audio, noise_factor=0.003)
        audio = change_pitch(audio, sr=sr, pitch_factor=np.random.uniform(-2, 2))
        audio = change_speed(audio, speed_factor=np.random.uniform(0.9, 1.1))
        return audio
    else:
        return audio
    
def generate_augmented_samples(audio, sr=SAMPLE_RATE, n_augmentations=3):
    """
    Generate multiple augmented versions of an audio sample.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        n_augmentations: Number of augmented versions to create
        
    Returns:
        list: List of augmented audio samples
    """
    augmented_samples = [audio] # include original

    augmentation_types = ["noise", "shift", "pitch", "speed", "reverb"]

    for i in range(n_augmentations):
        aug_type = augmentation_types[i % len(augmentation_types)]
        augmented = augment_audio(audio, sr=sr, augmentation_type=aug_type)
        augmented_samples.append(augmented)
    
    return augmented_samples

class DataAugmenter:
    """
    Class for handling data augmentation in the preprocessing pipeline.
    """
    def __init__(self, sr=SAMPLE_RATE):
        self.sr = sr
    
    def augment_dataset(self, df, n_augmentations=2, augment_emotions=None):
        """
        Augment entire dataset with specified augmentations.
        
        Args:
            df: DataFrame with file paths
            n_augmentations: Number of augmented versions per sample
            augment_emotions: List of emotions to augment (None = all)
            
        Returns:
            tuple: (augmented_features, augmented_labels)
        """
        from src.data_loader import load_audio_file
        from src.feature_extraction import extract_all_features
        from tqdm import tqdm

        features = []
        labels = []

        for i, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting data"):
            emotion = row["emotion"]

            # Check if emotion should be augmented
            if augment_emotions is not None and emotion not in augment_emotions:
                continue
            try:
                audio = load_audio_file(row["file_path"])

                if audio is not None:
                    # Generate augmented samples
                    augmented_samples = generate_augmented_samples(
                        audio, sr=self.sr, n_augmentations=n_augmentations
                    )
                
                    # Extract features from each augmented sample
                    for aug_audio in augmented_samples:
                        feature_vector = extract_all_features(aug_audio, sr=self.sr)
                        features.append(feature_vector)
                        labels.append(emotion)
            except Exception as e:
                print(f"\nError augmenting {row["file_path"]}: {e}")
        
        return np.array(features), np.array(labels)
    
