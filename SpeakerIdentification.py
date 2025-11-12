"""
Speaker Identification Module
Modular, importable version of SpeakerIdentification.py
Supports both CLI and Streamlit integration
"""

import os
import wave
import pickle
import numpy as np
import sounddevice as sd
import soundfile as sf
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture
import warnings

warnings.filterwarnings("ignore")

# ==================== CONFIGURATION ====================
ROOT = os.path.dirname(os.path.abspath(__file__))
TRAINING_FOLDER = os.path.join(ROOT, "training_set")
TESTING_FOLDER = os.path.join(ROOT, "testing_set")
MODELS_FOLDER = os.path.join(ROOT, "trained_models")
TRAIN_MANIFEST = os.path.join(ROOT, "training_set_addition.txt")
TEST_MANIFEST = os.path.join(ROOT, "testing_set_addition.txt")
UNKNOWN_THRESHOLD = -50  # ⭐ HARD-CODED THRESHOLD

# Create directories
os.makedirs(TRAINING_FOLDER, exist_ok=True)
os.makedirs(TESTING_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# ==================== UTILITY FUNCTIONS ====================


def calculate_delta(array):
    """Calculate delta (first derivative) of features"""
    rows, cols = array.shape
    deltas = np.zeros((rows, 20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            first = max(i - j, 0)
            second = min(i + j, rows - 1)
            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + 2 *
                     (array[index[1][0]] - array[index[1][1]])) / 10
    return deltas


def extract_features(audio, rate):
    """Extract MFCC features + delta from audio"""
    mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01,
                             20, nfft=1200, appendEnergy=True)
    mfcc_feature = preprocessing.scale(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature, delta))
    return combined


def get_audio_devices():
    """Get list of available input devices"""
    devices = sd.query_devices()
    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((i, device['name']))
    return input_devices


# ==================== RECORDING FUNCTIONS ====================

def record_audio_train(speaker_name, duration=10, device_index=None, sample_rate=44100):
    """
    Record training audio for a speaker

    Args:
        speaker_name: Name of speaker
        duration: Recording duration in seconds
        device_index: Audio device index (None for default)
        sample_rate: Sample rate in Hz

    Returns:
        Filepath of saved recording
    """
    try:
        print(f"🎤 Recording {duration}s for {speaker_name}...")

        # Record audio
        audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate,
                       channels=1, device=device_index, dtype='float32')
        sd.wait()

        # Generate unique filename
        existing = [f for f in os.listdir(TRAINING_FOLDER)
                    if f.startswith(f"{speaker_name}_") and f.endswith(".wav")]
        count = len(existing) + 1

        filename = f"{speaker_name}_{count}.wav"
        filepath = os.path.join(TRAINING_FOLDER, filename)

        # Save audio
        sf.write(filepath, audio, sample_rate)

        # Add to manifest
        with open(TRAIN_MANIFEST, 'a') as f:
            f.write(f"{filename}\n")

        print(f"✓ Saved: {filepath}")
        return filepath

    except Exception as e:
        print(f"✗ Error recording: {e}")
        return None


def record_audio_test(label="test", duration=10, device_index=None, sample_rate=44100):
    """
    Record test audio

    Args:
        label: Label for test recording
        duration: Recording duration in seconds
        device_index: Audio device index
        sample_rate: Sample rate in Hz

    Returns:
        Filepath of saved recording
    """
    try:
        print(f"🎤 Recording test audio ({label})...")

        # Record audio
        audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate,
                       channels=1, device=device_index, dtype='float32')
        sd.wait()

        # Generate unique filename
        existing = [f for f in os.listdir(TESTING_FOLDER)
                    if f.startswith(f"{label}_") and f.endswith(".wav")]
        count = len(existing)

        filename = f"{label}_test{count}.wav"
        filepath = os.path.join(TESTING_FOLDER, filename)

        # Save audio
        sf.write(filepath, audio, sample_rate)

        # Add to manifest
        with open(TEST_MANIFEST, 'a') as f:
            f.write(f"{filename}\n")

        print(f"✓ Saved: {filepath}")
        return filepath

    except Exception as e:
        print(f"✗ Error recording: {e}")
        return None


def save_uploaded_train(file_bytes, filename, speaker_name):
    """Save uploaded training file"""
    try:
        # Ensure unique filename
        base, ext = os.path.splitext(filename)
        filepath = os.path.join(TRAINING_FOLDER, f"{speaker_name}_{base}{ext}")

        with open(filepath, 'wb') as f:
            f.write(file_bytes)

        # Add to manifest
        with open(TRAIN_MANIFEST, 'a') as f:
            f.write(f"{os.path.basename(filepath)}\n")

        return filepath
    except Exception as e:
        print(f"✗ Error saving: {e}")
        return None


def save_uploaded_test(file_bytes, filename, label="test"):
    """Save uploaded test file"""
    try:
        base, ext = os.path.splitext(filename)
        filepath = os.path.join(TESTING_FOLDER, f"{label}_{base}{ext}")

        with open(filepath, 'wb') as f:
            f.write(file_bytes)

        # Add to manifest
        with open(TEST_MANIFEST, 'a') as f:
            f.write(f"{os.path.basename(filepath)}\n")

        return filepath
    except Exception as e:
        print(f"✗ Error saving: {e}")
        return None


# ==================== TRAINING FUNCTIONS ====================

def train_models(progress_callback=None):
    """
    Train GMM models for all speakers

    Args:
        progress_callback: Function to call with progress updates

    Returns:
        Training summary dict
    """
    if not os.path.exists(TRAIN_MANIFEST):
        return {"error": f"{TRAIN_MANIFEST} does not exist"}

    count = 1
    features = np.asarray(())
    summary = {"speakers": {}, "total_samples": 0}

    try:
        with open(TRAIN_MANIFEST, 'r') as file_paths:
            for path in file_paths:
                path = path.strip()
                if not path:
                    continue

                audio_path = os.path.join(TRAINING_FOLDER, path)

                if not os.path.exists(audio_path):
                    print(f"⚠️  File not found: {audio_path}")
                    continue

                try:
                    sr, audio = read(audio_path)
                    vector = extract_features(audio, sr)

                    if features.size == 0:
                        features = vector
                    else:
                        features = np.vstack((features, vector))

                    summary["total_samples"] += 1

                    if count == 5:
                        gmm = GaussianMixture(
                            n_components=6, max_iter=200,
                            covariance_type='diag', n_init=3)
                        gmm.fit(features)

                        speaker_name = path.split('_')[0]
                        picklefile = f"{speaker_name}.gmm"
                        model_path = os.path.join(MODELS_FOLDER, picklefile)

                        pickle.dump(gmm, open(model_path, 'wb'))

                        msg = f'✓ Modeling completed: {picklefile} ({features.shape})'
                        print(msg)

                        if progress_callback:
                            progress_callback(msg)

                        summary["speakers"][speaker_name] = {
                            "samples": features.shape[0],
                            "model_file": picklefile
                        }

                        features = np.asarray(())
                        count = 0

                    count += 1

                except Exception as e:
                    print(f"✗ Error processing {path}: {e}")
                    continue

    except Exception as e:
        return {"error": str(e)}

    return summary


# ==================== TESTING FUNCTIONS ====================

def test_model(audio_path, threshold=UNKNOWN_THRESHOLD):
    """
    Test single audio file against trained models

    Args:
        audio_path: Path to test audio
        threshold: Unknown speaker threshold

    Returns:
        Tuple of (predicted_speaker, score)
    """
    try:
        gmm_files = [os.path.join(MODELS_FOLDER, fname) for fname in
                     os.listdir(MODELS_FOLDER) if fname.endswith('.gmm')]

        if not gmm_files:
            return "No trained models", 0.0

        models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
        speakers = [os.path.splitext(os.path.basename(fname))[
            0] for fname in gmm_files]

        sr, audio = read(audio_path)
        vector = extract_features(audio, sr)

        if vector is None or vector.size == 0:
            return "Error", 0.0

        log_likelihood = np.zeros(len(models))
        for i, gmm in enumerate(models):
            try:
                scores = np.array(gmm.score(vector))
                log_likelihood[i] = scores.sum()
            except Exception as e:
                print(f"Error scoring with {speakers[i]}: {e}")
                log_likelihood[i] = float('-inf')

        winner = np.argmax(log_likelihood)
        max_score = log_likelihood[winner]

        if max_score < threshold:
            return "UNKNOWN SPEAKER", max_score

        return speakers[winner], max_score

    except Exception as e:
        print(f"Error testing: {e}")
        return "Error", 0.0


def test_all_from_list(threshold=UNKNOWN_THRESHOLD):
    """
    Test all files from testing_set_addition.txt

    Returns:
        List of (filename, predicted_speaker, score) tuples
    """
    results = []

    if not os.path.exists(TEST_MANIFEST):
        return results

    gmm_files = [os.path.join(MODELS_FOLDER, fname) for fname in
                 os.listdir(MODELS_FOLDER) if fname.endswith('.gmm')]

    if not gmm_files:
        return results

    models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
    speakers = [os.path.splitext(os.path.basename(fname))[0]
                for fname in gmm_files]

    with open(TEST_MANIFEST, 'r') as f:
        for filename in f:
            filename = filename.strip()
            if not filename:
                continue

            audio_path = os.path.join(TESTING_FOLDER, filename)

            if not os.path.exists(audio_path):
                results.append((filename, "File not found", 0.0))
                continue

            try:
                sr, audio = read(audio_path)
                vector = extract_features(audio, sr)

                if vector is None or vector.size == 0:
                    results.append((filename, "Error", 0.0))
                    continue

                log_likelihood = np.zeros(len(models))
                for i, gmm in enumerate(models):
                    try:
                        scores = np.array(gmm.score(vector))
                        log_likelihood[i] = scores.sum()
                    except:
                        log_likelihood[i] = float('-inf')

                winner = np.argmax(log_likelihood)
                max_score = log_likelihood[winner]

                if max_score < threshold:
                    results.append((filename, "UNKNOWN SPEAKER", max_score))
                else:
                    results.append((filename, speakers[winner], max_score))

            except Exception as e:
                results.append((filename, "Error", 0.0))

    return results


def get_training_files():
    """Get list of training files with metadata"""
    files = []

    if not os.path.exists(TRAIN_MANIFEST):
        return files

    with open(TRAIN_MANIFEST, 'r') as f:
        for filename in f:
            filename = filename.strip()
            if not filename:
                continue

            filepath = os.path.join(TRAINING_FOLDER, filename)
            if os.path.exists(filepath):
                speaker = filename.split('_')[0]
                size = os.path.getsize(filepath)
                files.append({
                    "filename": filename,
                    "speaker": speaker,
                    "filepath": filepath,
                    "size": size
                })

    return files


def get_testing_files():
    """Get list of testing files"""
    files = []

    if not os.path.exists(TEST_MANIFEST):
        return files

    with open(TEST_MANIFEST, 'r') as f:
        for filename in f:
            filename = filename.strip()
            if not filename:
                continue

            filepath = os.path.join(TESTING_FOLDER, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                files.append({
                    "filename": filename,
                    "filepath": filepath,
                    "size": size
                })

    return files


def delete_training_file(filename):
    """Delete training file"""
    filepath = os.path.join(TRAINING_FOLDER, filename)

    if os.path.exists(filepath):
        os.remove(filepath)

        # Remove from manifest
        with open(TRAIN_MANIFEST, 'r') as f:
            lines = f.readlines()

        with open(TRAIN_MANIFEST, 'w') as f:
            for line in lines:
                if line.strip() != filename:
                    f.write(line)

        return True
    return False


def delete_testing_file(filename):
    """Delete testing file"""
    filepath = os.path.join(TESTING_FOLDER, filename)

    if os.path.exists(filepath):
        os.remove(filepath)

        # Remove from manifest
        with open(TEST_MANIFEST, 'r') as f:
            lines = f.readlines()

        with open(TEST_MANIFEST, 'w') as f:
            for line in lines:
                if line.strip() != filename:
                    f.write(line)

        return True
    return False


def get_trained_speakers():
    """Get list of trained speakers"""
    speakers = []

    if os.path.exists(MODELS_FOLDER):
        for file in os.listdir(MODELS_FOLDER):
            if file.endswith('.gmm'):
                speakers.append(file.replace('.gmm', ''))

    return sorted(speakers)
