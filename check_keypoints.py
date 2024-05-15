import numpy as np
import os
from pathlib import Path

# Define the path where keypoints are stored
AUGMENTED_KEYPOINTS_PATH = "augmentation_keypoints/bread"  # Directory where augmented keypoints are stored

def check_keypoints(npy_path):
    keypoints = np.load(npy_path)
    if np.all(keypoints == 0):
        return True
    return False

def main():
    print("Checking for .npy files with all zero keypoints...")
    # Get all .npy files from the augmented keypoints directory
    npy_files = list(Path(AUGMENTED_KEYPOINTS_PATH).rglob("*.npy"))

    if not npy_files:
        print(f"No .npy files found in {AUGMENTED_KEYPOINTS_PATH}.")
        return

    all_zero_files = []

    # Loop through each .npy file
    for npy_file in npy_files:
        print(f"Checking {npy_file}...")
        if check_keypoints(npy_file):
            all_zero_files.append(npy_file)

    if all_zero_files:
        print("The following .npy files contain only zeros:")
        for file in all_zero_files:
            print(file)
    else:
        print("All .npy files contain valid keypoints.")

if __name__ == "__main__":
    main()