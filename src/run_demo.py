import os
import sys

PY = sys.executable  # uses your venv python

def run(cmd: str):
    # Always show what command is being executed
    print(f"\n▶ Running: {cmd}\n", flush=True)
    os.system(cmd)

def main():
    print("✅ run_demo started", flush=True)

    while True:
        print("\n🌿 AGROSENSE – FINAL DEMO MENU", flush=True)
        print("1 → Train ResNet50 (1 epoch demo)")
        print("2 → Train MobileNetV2 (1 epoch demo)")
        print("3 → Train EfficientNetB0 (1 epoch demo)")
        print("4 → Train DenseNet201 (1 epoch demo)")
        print("5 → Predict using ResNet50")
        print("6 → Predict using MobileNetV2")
        print("7 → Predict using EfficientNetB0")
        print("8 → Predict using DenseNet201")
        print("9 → Show performance graphs (from logs)")
        print("10 → Compare all models (quick test)")
        print("0 → Exit")

        choice = input("Select option: ").strip()

        if choice == "1":
            run(f'"{PY}" -m src.train_backbone --backbone resnet50 --epochs 1 --lr 1e-3')
        elif choice == "2":
            run(f'"{PY}" -m src.train_backbone --backbone mobilenetv2 --epochs 1 --lr 1e-3')
        elif choice == "3":
            run(f'"{PY}" -m src.train_backbone --backbone efficientnetb0 --epochs 1 --lr 1e-3')
        elif choice == "4":
            run(f'"{PY}" -m src.train_backbone --backbone densenet201 --epochs 1 --lr 1e-3')
        elif choice == "5":
            path = input("Enter FULL image path: ").strip()
            run(f'"{PY}" -m src.predict_backbone --backbone resnet50 --image "{path}"')
        elif choice == "6":
            path = input("Enter FULL image path: ").strip()
            run(f'"{PY}" -m src.predict_backbone --backbone mobilenetv2 --image "{path}"')
        elif choice == "7":
            path = input("Enter FULL image path: ").strip()
            run(f'"{PY}" -m src.predict_backbone --backbone efficientnetb0 --image "{path}"')
        elif choice == "8":
            path = input("Enter FULL image path: ").strip()
            run(f'"{PY}" -m src.predict_backbone --backbone densenet201 --image "{path}"')
        elif choice == "9":
            run(f'"{PY}" -m src.plot_history')
        elif choice == "10":
            run(f'"{PY}" -m src.demo_all_models')
        elif choice == "0":
            print("Bye ✅", flush=True)
            break
        else:
            print("Invalid option. Try again.", flush=True)

if __name__ == "__main__":
    main()
