import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment

# Load MP3 file and convert it to WAV (Librosa does not support MP3 directly)
def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

# Extract MFCC features from WAV file
def extract_mfcc(wav_path):
    y, sr = librosa.load(wav_path, sr=22050)  # Load audio
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCC features
    return mfccs.T  # Transpose to get time-step-wise data

# Save extracted features to CSV
def save_to_csv(mfccs, csv_path):
    df = pd.DataFrame(mfccs)
    df.to_csv(csv_path, index=False)

# Convert MP3 to WAV and extract features
mp3_path = "audio.mp3"  # Input MP3 file
wav_path = "audio.wav"  # Temporary WAV file
csv_path = "audio_features.csv"  # Output CSV file

convert_mp3_to_wav(mp3_path, wav_path)  # Convert MP3 to WAV
mfcc_features = extract_mfcc(wav_path)  # Extract features
save_to_csv(mfcc_features, csv_path)  # Save to CSV

print(f"CSV file saved: {csv_path}")
