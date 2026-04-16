"""
batch_sketch2real.py — Convert a folder of sketches to real images

Usage:
    python batch_sketch2real.py --folder ./sketches --prompt "a house, photorealistic"
    python batch_sketch2real.py --folder ./sketches --prompts_file prompts.txt
    python batch_sketch2real.py --folder ./sketches --prompt "cityscape" --out_folder ./results
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

SUPPORTED = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_pipeline(device, low_vram):
    import torch
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

    print("[loading] ControlNet scribble...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    print("[loading] Stable Diffusion 1.5...")
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
    else:
        pipe = pipe.to(device)

    return pipe


def preprocess_sketch(image_path, size=512):
    import cv2
    import numpy as np
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))

    arr = np.array(canvas)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if gray.mean() > 128:
        gray = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    return Image.fromarray(binary).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Batch sketch to real image conversion")
    parser.add_argument("--folder",       required=True, help="Folder containing sketch images")
    parser.add_argument("--prompt",       default="photorealistic, high quality, detailed", help="Single prompt for all images")
    parser.add_argument("--prompts_file", default=None, help="Text file with one prompt per line (matched by filename order)")
    parser.add_argument("--negative",     default="blurry, low quality, ugly, deformed, watermark", help="Negative prompt")
    parser.add_argument("--out_folder",   default=None, help="Output folder (default: <folder>/results)")
    parser.add_argument("--steps",        type=int,   default=25)
    parser.add_argument("--guidance",     type=float, default=7.5)
    parser.add_argument("--strength",     type=float, default=0.9)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--size",         type=int,   default=512)
    parser.add_argument("--device",       default=None)
    parser.add_argument("--low-vram",     action="store_true")
    args = parser.parse_args()

    # Collect input files
    sketches = sorted([
        os.path.join(args.folder, f)
        for f in os.listdir(args.folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED
    ])

    if not sketches:
        print(f"[ERROR] No image files found in {args.folder}")
        sys.exit(1)

    print(f"Found {len(sketches)} sketch(es) in {args.folder}")

    # Output folder
    out_folder = args.out_folder or os.path.join(args.folder, "results")
    os.makedirs(out_folder, exist_ok=True)
    print(f"Output folder: {out_folder}\n")

    # Load prompts
    prompts = []
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    # Pad prompts with default if fewer than sketches
    while len(prompts) < len(sketches):
        prompts.append(args.prompt)

    # Detect device
    if args.device:
        device = args.device
        low_vram = args.low_vram
    else:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                low_vram = vram < 6
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                low_vram = False
            else:
                device = "cpu"
                low_vram = False
        except Exception:
            device = "cpu"
            low_vram = False

    print(f"Device: {device}")

    # Load pipeline once
    pipe = load_pipeline(device, low_vram)

    import torch
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Process each sketch
    for i, sketch_path in enumerate(sketches):
        stem = os.path.splitext(os.path.basename(sketch_path))[0]
        out_path = os.path.join(out_folder, f"{stem}_real.png")
        prompt = prompts[i]

        print(f"\n[{i+1}/{len(sketches)}] {os.path.basename(sketch_path)}")
        print(f"  prompt: {prompt}")

        try:
            scribble = preprocess_sketch(sketch_path, size=args.size)
            result = pipe(
                prompt=prompt,
                negative_prompt=args.negative,
                image=scribble,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                controlnet_conditioning_scale=args.strength,
                generator=generator,
            )
            result.images[0].save(out_path)
            print(f"  saved: {out_path}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

    print(f"\nBatch complete. {len(sketches)} image(s) processed → {out_folder}")


if __name__ == "__main__":
    main()
