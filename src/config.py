from pathlib import Path

# Base dataset directory
DATASET_BASE = Path("../data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)")

TRAIN_DIR = DATASET_BASE / "train"
VALID_DIR = DATASET_BASE / "valid"
TEST_DIR = Path("../data/raw/test/test")  # optional (we will use later)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

