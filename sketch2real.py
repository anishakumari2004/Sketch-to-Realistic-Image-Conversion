"""
sketch2real.py — Convert any sketch image to a realistic photo
Uses: Stable Diffusion 1.5 + ControlNet Scribble (local, no API)

Usage:
    python sketch2real.py --sketch path/to/sketch.png --prompt "a cat sitting"
    python sketch2real.py --sketch sketch.png --prompt "a house" --steps 30 --strength 0.9
    python sketch2real.py --sketch sketch.png  # uses auto prompt
"""

import argparse
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def check_dependencies():
    missing = []
    for pkg in ["torch", "diffusers", "transformers", "accelerate", "PIL", "cv2", "numpy"]:
        try:
            __import__(pkg if pkg != "PIL" else "PIL.Image")
        except ImportError:
            missing.append(pkg if pkg != "PIL" else "Pillow")
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)


def load_pipeline(device: str, low_vram: bool):
    """Load ControlNet + SD pipeline. Downloads weights on first run (~5GB)."""
    import torch
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

    print("[1/3] Loading ControlNet scribble model...")
    print("      (First run downloads ~1.5GB weights — wait ~5 min)")

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    print("[2/3] Loading Stable Diffusion 1.5...")
    print("      (First run downloads ~4GB weights — wait ~10 min)")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if low_vram:
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        print("      [low VRAM mode enabled]")
    else:
        pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()

    print("[3/3] Pipeline ready.")
    return pipe


def preprocess_sketch(image_path: str, size: int = 512):
    """
    Preprocess sketch image:
    - Resize to square (model expects 512x512 by default)
    - Convert to scribble format (white lines on black background)
    - If input is already white-on-black, it's used as-is
    - If input is black-on-white (typical pencil sketch), it's inverted
    """
    import cv2
    import numpy as np
    from PIL import Image

    img = Image.open(image_path).convert("RGB")

    # Resize keeping aspect, pad to square
    img.thumbnail((size, size), Image.LANCZOS)
    new_img = Image.new("RGB", (size, size), (0, 0, 0))
    offset = ((size - img.width) // 2, (size - img.height) // 2)
    new_img.paste(img, offset)

    arr = np.array(new_img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Detect if sketch is dark-on-light (normal pencil sketch) → invert
    mean_val = gray.mean()
    if mean_val > 128:
        gray = cv2.bitwise_not(gray)

    # Threshold to clean binary scribble
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Convert back to RGB PIL image
    scribble = Image.fromarray(binary).convert("RGB")
    return scribble


def generate(
    sketch_path: str,
    prompt: str,
    negative_prompt: str,
    output_path: str,
    steps: int,
    guidance: float,
    controlnet_strength: float,
    seed: int,
    device: str,
    low_vram: bool,
    size: int,
):
    import torch
    from PIL import Image

    # Validate input
    if not os.path.exists(sketch_path):
        print(f"[ERROR] Sketch file not found: {sketch_path}")
        sys.exit(1)

    print(f"\n  Sketch     : {sketch_path}")
    print(f"  Prompt     : {prompt}")
    print(f"  Output     : {output_path}")
    print(f"  Device     : {device}")
    print(f"  Steps      : {steps}")
    print(f"  Seed       : {seed if seed != -1 else 'random'}\n")

    # Preprocess sketch
    scribble_img = preprocess_sketch(sketch_path, size=size)
    scribble_preview = output_path.replace(".png", "_scribble_preview.png")
    scribble_img.save(scribble_preview)
    print(f"  Scribble preview saved to: {scribble_preview}")

    # Load model
    pipe = load_pipeline(device, low_vram)

    # Set seed
    generator = None
    if seed != -1:
        generator = torch.Generator(device=device).manual_seed(seed)

    # Run inference
    print("\n  Generating image...")
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=scribble_img,
        num_inference_steps=steps,
        guidance_scale=guidance,
        controlnet_conditioning_scale=controlnet_strength,
        generator=generator,
    )

    out_img = result.images[0]
    out_img.save(output_path)
    print(f"\n  Done! Output saved to: {output_path}")
    return output_path


def auto_detect_device():
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU detected: {gpu} ({vram:.1f} GB VRAM)")
            return "cuda", vram < 6
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("  Apple Silicon MPS detected")
            return "mps", False
        else:
            print("  No GPU found — using CPU (slow, ~10-30 min per image)")
            return "cpu", False
    except Exception:
        return "cpu", False


def main():
    parser = argparse.ArgumentParser(
        description="Convert sketch images to realistic photos using ControlNet + SD1.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sketch2real.py --sketch my_sketch.png --prompt "a cat sitting on a couch"
  python sketch2real.py --sketch face.jpg --prompt "portrait of a woman, studio lighting, photorealistic"
  python sketch2real.py --sketch building.png --prompt "modern building exterior, photography" --steps 30
  python sketch2real.py --sketch sketch.png --prompt "landscape" --seed 42 --output result.png
        """,
    )

    parser.add_argument("--sketch",    required=True,  help="Path to input sketch image (jpg/png/webp)")
    parser.add_argument("--prompt",    default="a realistic photo, high quality, detailed, 8k",
                        help="Text prompt describing what to generate")
    parser.add_argument("--negative",  default="blurry, low quality, ugly, deformed, artifacts, watermark, text",
                        help="Negative prompt (what to avoid)")
    parser.add_argument("--output",    default=None,   help="Output image path (default: <sketch>_output.png)")
    parser.add_argument("--steps",     type=int,   default=25,   help="Inference steps (default: 25, more = better quality)")
    parser.add_argument("--guidance",  type=float, default=7.5,  help="Prompt guidance scale (default: 7.5)")
    parser.add_argument("--strength",  type=float, default=0.9,  help="ControlNet strength 0.0-2.0 (default: 0.9)")
    parser.add_argument("--seed",      type=int,   default=-1,   help="Random seed for reproducibility (-1 = random)")
    parser.add_argument("--size",      type=int,   default=512,  help="Output size in pixels (default: 512, use 768 for higher res)")
    parser.add_argument("--device",    default=None,   help="Force device: cuda / mps / cpu (auto-detected if not set)")
    parser.add_argument("--low-vram",  action="store_true",      help="Enable low VRAM mode (for GPUs with <6GB)")

    args = parser.parse_args()

    print("\n=== sketch2real ===\n")
    check_dependencies()

    # Auto-detect device
    if args.device is None:
        device, low_vram = auto_detect_device()
    else:
        device = args.device
        low_vram = args.low_vram

    # Auto-set output path
    output = args.output
    if output is None:
        stem = Path(args.sketch).stem
        output = f"{stem}_output.png"

    generate(
        sketch_path=args.sketch,
        prompt=args.prompt,
        negative_prompt=args.negative,
        output_path=output,
        steps=args.steps,
        guidance=args.guidance,
        controlnet_strength=args.strength,
        seed=args.seed,
        device=device,
        low_vram=low_vram,
        size=args.size,
    )


if __name__ == "__main__":
    main()
