import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

LOG_DIR = Path("models/logs")
PLOT_DIR = Path("models/plots")
PLOT_DIR.mkdir(exist_ok=True)

files = list(LOG_DIR.glob("*stage1_log.csv"))

if not files:
    print("No log files found.")
    exit()

for file in files:
    name = file.stem.replace("_stage1_log", "")
    df = pd.read_csv(file)

    plt.figure(figsize=(8,5))
    plt.plot(df["accuracy"], label="train_acc")
    plt.plot(df["val_accuracy"], label="val_acc")
    plt.title(f"{name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    save_path = PLOT_DIR / f"{name}_accuracy.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Saved plot → {save_path}")

print("\nAll model plots saved in models/plots/")
