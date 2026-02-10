from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_BASE = ROOT_DIR / "data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"

TRAIN_DIR = DATASET_BASE / "train"
VALID_DIR = DATASET_BASE / "valid"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

EPOCHS_STAGE1 = 3
