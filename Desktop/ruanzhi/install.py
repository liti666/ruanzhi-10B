"""
Run this instead of `pip install -r requirements.txt`.
Detects GPU and installs the correct PyTorch build automatically.

Usage:
    python install.py
"""
import subprocess
import sys


def run(cmd):
    print(f">>> {' '.join(cmd)}")
    subprocess.check_call(cmd)


def has_nvidia_gpu():
    try:
        subprocess.check_output("nvidia-smi", stderr=subprocess.DEVNULL)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def main():
    pip = [sys.executable, "-m", "pip", "install"]

    # Step 1: Install PyTorch with the right build
    print("\n[1/2] Detecting GPU...")
    if has_nvidia_gpu():
        print("  NVIDIA GPU detected -> installing PyTorch CUDA 11.8")
        run(pip + [
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ])
    else:
        print("  No GPU detected -> installing PyTorch CPU")
        run(pip + ["torch", "torchvision", "torchaudio"])

    # Step 2: Install remaining packages (torch already pinned, won't be overwritten)
    print("\n[2/2] Installing other dependencies...")
    run(pip + [
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "textattack>=0.3.8",
        "scikit-learn>=1.2.0",
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "tqdm>=4.64.0",
        "nltk>=3.8.0",
        "sentence-transformers>=2.2.0",
    ])

    # Verify
    print("\nVerifying installation...")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"  torch {torch.__version__}  |  CUDA: {cuda_ok}", end="")
        if cuda_ok:
            print(f"  |  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print()
        print("  Install complete.")
    except ImportError as e:
        print(f"  Verification failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
