#!/usr/bin/env python3
"""
setup_check.py — Run this once before using sketch2real.py
Checks your system and installs dependencies.
"""

import sys
import subprocess
import platform


def run(cmd):
    print(f"  > {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] {result.stderr.strip()}")
    else:
        print(f"  OK")
    return result.returncode == 0


def check_python():
    v = sys.version_info
    print(f"Python {v.major}.{v.minor}.{v.micro}", end=" ")
    if v.major < 3 or (v.major == 3 and v.minor < 8):
        print("— [ERROR] Need Python 3.8+")
        sys.exit(1)
    print("— OK")


def check_torch_gpu():
    try:
        import torch
        print(f"PyTorch {torch.__version__}", end=" ")
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"— CUDA GPU: {name} ({vram:.1f} GB VRAM)")
            if vram < 4:
                print("  [WARN] <4GB VRAM — use --low-vram flag, CPU fallback")
            elif vram < 6:
                print("  [WARN] <6GB VRAM — use --low-vram flag")
            else:
                print(f"  GPU OK — can run without --low-vram")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("— Apple Silicon MPS (OK)")
        else:
            print("— CPU only (will be slow ~10-30 min/image)")
        return True
    except ImportError:
        print("— NOT INSTALLED")
        return False


def install_deps():
    print("\nInstalling dependencies...")
    
    # Try CUDA torch first, fallback to CPU
    try:
        import torch
        print("torch already installed, skipping.")
    except ImportError:
        system = platform.system()
        print("Installing PyTorch...")
        if system == "Linux" or system == "Windows":
            run("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        elif system == "Darwin":
            run("pip install torch torchvision")

    # Install other deps
    run("pip install diffusers>=0.25.0 transformers>=4.36.0 accelerate>=0.25.0")
    run("pip install Pillow>=10.0.0 opencv-python>=4.8.0 numpy>=1.24.0 tqdm")


def check_all_imports():
    print("\nChecking all imports...")
    pkgs = {
        "torch": "torch",
        "diffusers": "diffusers",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "PIL": "Pillow",
        "cv2": "opencv-python",
        "numpy": "numpy",
    }
    all_ok = True
    for mod, pkg in pkgs.items():
        try:
            __import__(mod)
            print(f"  {pkg} — OK")
        except ImportError:
            print(f"  {pkg} — MISSING (run: pip install {pkg})")
            all_ok = False
    return all_ok


def main():
    print("=" * 50)
    print("  sketch2real — system check")
    print("=" * 50)

    print("\n[1] Python version")
    check_python()

    print("\n[2] GPU / PyTorch check")
    torch_ok = check_torch_gpu()

    if not torch_ok:
        print("\nPyTorch not found. Installing now...")
        install_deps()

    ok = check_all_imports()

    print("\n" + "=" * 50)
    if ok:
        print("  All good! You can now run:")
        print("  python sketch2real.py --sketch your_sketch.png --prompt 'a cat'")
    else:
        print("  Some packages missing. Run:")
        print("  pip install -r requirements.txt")
    print("=" * 50)


if __name__ == "__main__":
    main()
