import os
import shutil
import random
import numpy as np

import tensorflow as tf

# ------------------------------------------------------------
# Step 1: Basic Configuration
# ------------------------------------------------------------

print("=" * 70)
print("                   Marvellous Infosystems")
print("        Industrial Surface Crack Detection using CNN")
print("=" * 70)

ORIGINAL_DATASET = "CrackDataset"

POSITIVE_FOLDER = os.path.join(ORIGINAL_DATASET, "Positive")
NEGATIVE_FOLDER = os.path.join(ORIGINAL_DATASET, "Negative")

PROCESSED_DATASET = "Processed_CrackDataset"

# Standard input size for every image supplied to the CNN.
IMAGE_SIZE = 128
# Number of images processed in one training batch.
BATCH_SIZE = 32
# Maximum number of complete passes through the training dataset.
EPOCHS = 15
# Fixed seed improves reproducibility of random operations.
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ------------------------------------------------------------
# Step 2: Check Original Dataset
# ------------------------------------------------------------

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        print("ERROR: Folder not found:", folder_path)
        exit()


check_folder(POSITIVE_FOLDER)
check_folder(NEGATIVE_FOLDER)


def get_image_files(folder):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return [
        file for file in os.listdir(folder)
        if file.lower().endswith(valid_extensions)
    ]


positive_images = get_image_files(POSITIVE_FOLDER)
negative_images = get_image_files(NEGATIVE_FOLDER)

print("Original Positive Images:", len(positive_images))
print("Original Negative Images:", len(negative_images))

if len(positive_images) == 0 or len(negative_images) == 0:
    print("ERROR: Positive or Negative folder contains no images.")
    exit()


# ------------------------------------------------------------
# Step 3: Create Industrial Folder Structure
# ------------------------------------------------------------

folders = [
    "train/Crack",
    "train/NoCrack",
    "validation/Crack",
    "validation/NoCrack",
    "test/Crack",
    "test/NoCrack"
]

# Remove any previously generated split so old images cannot remain in
# train/validation/test folders when the program is executed again.
if os.path.exists(PROCESSED_DATASET):
    shutil.rmtree(PROCESSED_DATASET)

# Create a fresh train/validation/test directory structure.
for folder in folders:
    os.makedirs(os.path.join(PROCESSED_DATASET, folder), exist_ok=True)
