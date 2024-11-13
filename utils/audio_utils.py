import librosa
import numpy as np
import random
# from sklearn.preprocessing import StandardScaler

def load_audio(file_path, sr=16000):
    """
    Load audio file using librosa
    Args:
        file_path: Path to audio file
        sr: Target sampling rate (default: 16000 Hz)
    Returns:
        audio: Audio signal, sampling rate
    """
    audio, sr = librosa.load(file_path, sr=sr)
    return audio, sr


def add_background_noise(speech_signal, noise_signal, snr_db):
    """
    Add background noise to speech at specified SNR
    Args:
        speech_signal: Speech audio signal
        noise_signal: Noise audio signal
        snr_db: Target Signal-to-Noise Ratio in dB
    Returns:
        noisy_speech: Speech with added background noise
    """
    # Calculate RMS energy for both signals
    speech_rms = np.sqrt(np.mean(speech_signal**2))
    noise_rms = np.sqrt(np.mean(noise_signal**2))

    # Calculate noise scaling factor for target SNR
    scaling_factor = speech_rms / (noise_rms * (10 ** (snr_db / 20)))
    scaled_noise = noise_signal * scaling_factor

    # Handle different lengths
    if len(scaled_noise) < len(speech_signal):
        # Pad noise by repetition
        scaled_noise = np.pad(
            scaled_noise, (0, len(speech_signal) - len(scaled_noise)), mode="wrap"
        )
    else:
        # Randomly crop noise
        start = random.randint(0, len(scaled_noise) - len(speech_signal))
        scaled_noise = scaled_noise[start : start + len(speech_signal)]

    # Mix signals
    noisy_speech = speech_signal + scaled_noise

    # Normalize to prevent clipping
    noisy_speech = librosa.util.normalize(noisy_speech)

    return noisy_speech


# def extract_mfcc(audio_path, n_mfcc=40, duration=1):
#     """
#     Extract MFCC features from an audio file
#     Parameters:
#     audio_path: str, path to audio file
#     sr: int, sampling rate in Hz (default: 16000 Hz)
#     n_mfcc: int, number of MFCC coefficients to extract (default: 13)
#     duration: float, duration in seconds to process (default: 3 seconds)
#             For example: duration=3 processes 3 seconds of audio
#             duration=0.5 processes 500 milliseconds
#     Returns:
#         mfcc: MFCC features
#     """
#     try:
#         # Load audio file
#         audio, sr = librosa.load(audio_path, sr=16000)

#         # # Ensure consistent length by padding or truncation
#         # target_legnth = int(duration * sr)

#         # if len(audio) < target_legnth:
#         #     audio = np.pad(audio, (0, target_legnth - len(audio)))
#         # else:
#         #     audio = audio[:target_legnth]

#         # Extract MFCC features
#         mfcc = librosa.feature.mfcc(
#             y=audio,
#             sr=sr,
#             n_mfcc=n_mfcc)
#         # mfcc = librosa.util.normalize(mfcc)
#         delta_mfcc = librosa.feature.delta(mfcc)
#         delta2_mfcc = librosa.feature.delta(mfcc, order=2)

#         mfcc = librosa.util.normalize(mfcc)
#         # delta_mfcc = librosa.util.normalize(delta_mfcc)
#         # delta2_mfcc = librosa.util.normalize(delta2_mfcc)

#         # Stack all features
#         # mfcc_features = np.concatenate([mfcc, delta_mfcc, delta2_mfcc])

#         return mfcc
#     except Exception as e:
#         print(f"Error processing {audio_path}: {str(e)}")
#         return None

def extract_mfcc(audio_path, n_mfcc=40):
    """
    Extract MFCC features from an audio file
    Parameters:
    audio_path: str, path to audio file
    Returns:
        mfcc: MFCC features
    """
    try:
        # Load audio file
        audio, sr = librosa.load(audio_path)

        # Normalized Audio between -1 and 1
        normalized_audio = librosa.util.normalize(audio)

        # Extraction of MFCC
        mfcc = librosa.feature.mfcc(y=normalized_audio, sr=sr, n_mfcc=n_mfcc)

        # Normalize MFCC between -1 and 1
        normalized_mfcc = librosa.util.normalize(mfcc)

        return normalized_mfcc
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None


# Given an numpy array of features, zero-pads each ocurrence to max_padding
def add_padding(features, mfcc_max_padding=44):
    padded = []

    # Add padding
    for i in range(len(features)):
        px = features[i]
        size = len(px[0])
        # Add padding if required
        if size < mfcc_max_padding:
            xDiff = mfcc_max_padding - size
            xLeft = xDiff // 2
            xRight = xDiff - xLeft
            px = np.pad(px, pad_width=((0, 0), (xLeft, xRight)), mode="constant")

        padded.append(px)

    return padded
