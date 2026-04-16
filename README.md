# sketch2real

Convert any hand-drawn sketch or line art into a realistic photo.
Runs **100% locally** — no API, no internet needed after first model download.

---

## What it uses

| Component | Model | Size |
|-----------|-------|------|
| Image structure | ControlNet Scribble (lllyasviel) | ~1.5 GB |
| Image generation | Stable Diffusion 1.5 (RunwayML) | ~4 GB |
| Total download | First run only | ~5.5 GB |

---

## Requirements

- Python 3.8+
- 8 GB RAM minimum (16 GB recommended)
- GPU: NVIDIA 6GB+ VRAM recommended (works on CPU too, just slow)
- macOS Apple Silicon: MPS backend auto-used

---

## Setup (one time)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Check your system
python setup_check.py
```

---

## Usage

### Single image

```bash
python sketch2real.py --sketch my_sketch.png --prompt "a cat sitting on a couch"
```

### With more options

```bash
python sketch2real.py \
  --sketch face.jpg \
  --prompt "portrait of a woman, studio lighting, photorealistic, 8k" \
  --negative "blurry, cartoon, ugly" \
  --steps 30 \
  --strength 0.9 \
  --seed 42 \
  --output my_result.png
```

### Batch (whole folder)

```bash
python batch_sketch2real.py --folder ./my_sketches --prompt "architecture render, photorealistic"
```

### With per-image prompts (one per line in a .txt file)

```bash
python batch_sketch2real.py --folder ./sketches --prompts_file prompts.txt
```

---

## All arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--sketch` | required | Path to input sketch (jpg/png/webp) |
| `--prompt` | auto | Describe what to generate |
| `--negative` | auto | What to avoid in output |
| `--output` | `<sketch>_output.png` | Output file path |
| `--steps` | `25` | More steps = better quality (20–50 range) |
| `--guidance` | `7.5` | How closely to follow prompt (5–12) |
| `--strength` | `0.9` | ControlNet influence (0.5–1.5) |
| `--seed` | `-1` (random) | Fix seed for reproducible results |
| `--size` | `512` | Output resolution (512 or 768) |
| `--device` | auto | Force: `cuda` / `mps` / `cpu` |
| `--low-vram` | off | Enable for GPUs with <6GB VRAM |

---

## Tips for best results

**Prompts:**
- Be specific: `"a tabby cat sitting, photorealistic, studio lighting"` > `"cat"`
- Add quality boosters: `"8k, photorealistic, highly detailed, sharp focus"`
- Specify style: `"oil painting"`, `"watercolor"`, `"architectural render"`, `"portrait photography"`

**Sketch quality:**
- Clean, clear outlines work best
- Both dark-on-white and white-on-dark sketches supported (auto-detected)
- Higher contrast → better structure following

**Parameters:**
- `--strength 0.6–0.8` → more creative, loosely follows sketch
- `--strength 1.0–1.5` → strictly follows sketch structure
- `--steps 30–50` → higher quality but slower
- `--guidance 9–12` → follows prompt more strictly

**Low VRAM (under 6GB GPU):**
```bash
python sketch2real.py --sketch sketch.png --prompt "a house" --low-vram
```

**CPU only (slow but works):**
- Expect 10–30 minutes per image
- Use `--steps 15` to speed up

---

## Project structure

```
sketch2real/
├── sketch2real.py          # Main script (single image)
├── batch_sketch2real.py    # Batch processing (folder of images)
├── setup_check.py          # First-time system check
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## How it works

1. Your sketch is preprocessed into a **scribble map** (white lines on black)
2. ControlNet reads the scribble to understand the **spatial structure**
3. Stable Diffusion generates a realistic image **guided by your prompt**
4. ControlNet keeps the output **anchored to your sketch's shapes**

The result follows your sketch's composition but has realistic textures, lighting, and detail.

---

## Troubleshooting

**Out of memory (CUDA OOM):**
```bash
python sketch2real.py --sketch s.png --prompt "..." --low-vram --size 512
```

**Slow on CPU:**
```bash
python sketch2real.py --sketch s.png --prompt "..." --steps 15
```

**Output doesn't match sketch:**
- Increase `--strength` (try 1.2)
- Use cleaner/simpler sketch

**Output looks bad / noisy:**
- Increase `--steps` to 30–40
- Improve your prompt
- Try different `--seed` values
