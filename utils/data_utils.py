import os
from pathlib import Path
import soundfile as sf
import random
from .audio_utils import *
import numpy as np
import matplotlib.pyplot as plt



def augment_dataset(
    speech_folder,
    noise_folder,
    output_folder,
    sr=16000,
    snr_range=(5, 15),
    num_augmentations=1,
):
    """
    Augment dataset while preserving directory structure and including original files
    Args:
        speech_folder: Root folder containing speech files in subdirectories
        noise_folder: Folder containing noise files
        output_folder: Root folder to save augmented files
        sr: Target sampling rate
        snr_range: Range of Signal-to-Noise Ratio in dB (min, max)
        num_augmentations: Number of noisy versions to create for each file
    """
    # Convert paths to Path objects
    speech_folder = Path(speech_folder)
    noise_folder = Path(noise_folder)
    output_folder = Path(output_folder)

    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load all noise files
    print("Loading noise files...")
    noise_files = []
    for noise_file in noise_folder.glob("*"):
        if noise_file.suffix.lower() in [".wav", ".mp3", ".flac"]:
            noise, _ = load_audio(str(noise_file), sr=sr)
            noise_files.append(noise)

    if not noise_files:
        raise ValueError("No noise files found in the noise folder!")

    # Get all audio files while preserving directory structure
    audio_files = []
    for ext in [".wav", ".mp3", ".flac"]:
        audio_files.extend(speech_folder.rglob(f"*{ext}"))

    # Process each speech file
    total_files = len(audio_files)
    for idx, speech_path in enumerate(audio_files, 1):
        # Get relative path to preserve directory structure
        rel_path = speech_path.relative_to(speech_folder)

        # Create output subdirectory
        output_subdir = output_folder / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {idx}/{total_files}: {rel_path}")

        # Load and save original file (resampled to target sr if necessary)
        speech, sr = load_audio(str(speech_path), sr=sr)  # type: ignore
        original_output_path = output_subdir / f"original_{speech_path.name}"
        sf.write(str(original_output_path), speech, sr)

        # Create augmented versions
        for i in range(num_augmentations):
            # Randomly select noise and SNR
            noise = random.choice(noise_files)
            snr = random.uniform(snr_range[0], snr_range[1])

            # Add noise
            noisy_speech = add_background_noise(speech, noise, snr)

            # Create output filename
            output_filename = f"noisy_{i+1}_{speech_path.name}"
            output_path = output_subdir / output_filename

            # Save augmented audio
            sf.write(str(output_path), noisy_speech, sr)


def prepare_dateset(data_dir):
    """
    Prepare dataset from directory
    Args:
        data_dir: Directory containing audio files in class subdirectories
    Returns:
        X: MFCC features
        y: Labels
        labels: return lables as an array
    """
    data_dir = Path(data_dir)
    features = []
    labels = []
    max_frames = 0
    frames_num = 0

    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        class_name = class_dir.name
        audio_files = list(class_dir.glob("*.wav"))
        print(f"Processing class: {class_name} - {len(audio_files)} files")
        for audio_file in audio_files:
            mfcc = extract_mfcc(str(audio_file))
            if mfcc is not None:
                features.append(mfcc)
                labels.append(class_name)
                frames_num = mfcc.shape[1]

            if frames_num > max_frames:
                max_frames = frames_num
    print(max_frames)
    padded_features = add_padding(features, max_frames)

    # Return the padded fatures and labels (The labeles are not one-hot encoded)
    return padded_features, labels


def verify_directory_structure(original_folder, augmented_folder):
    """
    Verify that the directory structure was preserved
    Args:
        original_folder: Original dataset root folder
        augmented_folder: Augmented dataset root folder
    Returns:
        bool: True if directory structure matches
    """
    original_dirs = set(
        str(p.relative_to(original_folder))
        for p in Path(original_folder).rglob("*")
        if p.is_dir()
    )
    augmented_dirs = set(
        str(p.relative_to(augmented_folder))
        for p in Path(augmented_folder).rglob("*")
        if p.is_dir()
    )

    return original_dirs == augmented_dirs


def create_dataset_summary(output_folder):
    """
    Create a summary of the dataset including file counts
    Args:
        output_folder: Root folder of the augmented dataset
    """
    output_folder = Path(output_folder)

    # Count files by type
    original_count = len(list(output_folder.rglob("original_*.wav")))
    augmented_count = len(list(output_folder.rglob("noisy_*.wav")))

    # Count directories
    subdirs = len([x for x in output_folder.rglob("*") if x.is_dir()])

    print("\nDataset Summary:")
    print(f"Total subdirectories: {subdirs}")
    print(f"Original files: {original_count}")
    print(f"Augmented files: {augmented_count}")
    print(f"Total files: {original_count + augmented_count}")

