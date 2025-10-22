# run_step1_3.py – ESREAL loop with BLIP-2 (LAVIS)

import json, argparse, os, sys
from pathlib import Path
from PIL import Image

import torch
from diffusers import StableDiffusionPipeline


# Add lavis to sys.path
lavis_path = os.path.join(os.path.dirname(__file__), "libs", "LAVIS")
if lavis_path not in sys.path:
    sys.path.insert(0, lavis_path)

from lavis.models import load_model_and_preprocess  # BLIP-2 helper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_image(pipe, prompt, out_path, steps=30, cfg=7.5, seed=0):
    g = torch.Generator(pipe.device).manual_seed(seed)
    img = pipe(prompt,
               num_inference_steps=steps,
               guidance_scale=cfg,
               generator=g).images[0]
    img.save(out_path)
    return img

def caption_image(blip_model, vis_proc, txt_proc, pil_img):
    sample = {
        "image": vis_proc(pil_img).unsqueeze(0).to(DEVICE),
        "text_input": [""],  # empty prompt for BLIP-2
    }
    with torch.no_grad():
        caption = blip_model.generate(sample, use_nucleus_sampling=False)[0]
    return caption

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # --- 1️⃣ Prompt → Original Image ---
    print("🔄 Loading Stable Diffusion XL …")
    sd = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",  # or "stabilityai/stable-diffusion-2-1"
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)


    original_path = out_dir / "original.png"
    print(f"🎨 Generating image from prompt: “{args.prompt}”")
    original_img = generate_image(sd, args.prompt, original_path, seed=args.seed)

    # --- 2️⃣ Original Image → Caption ---
    print("🧠 Loading BLIP-2 via LAVIS …")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=DEVICE
    )

    caption = caption_image(model, vis_processors["eval"], txt_processors["eval"], original_img)
    caption_path = out_dir / "caption.txt"
    caption_path.write_text(caption)
    print("📝 Caption:", caption)

    # --- 3️⃣ Caption → Regenerated Image ---
    regen_path = out_dir / "regen.png"
    print("🎨 Regenerating image from caption …")
    _ = generate_image(sd, caption, regen_path, seed=args.seed + 1)

    # --- 4️⃣ Save Outputs ---
    handoff = {
        "original": str(original_path),
        "caption": str(caption_path),
        "regen": str(regen_path)
    }
    json_path = out_dir / "handoff.json"
    json_path.write_text(json.dumps(handoff, indent=2))
    print("✅ Wrote handoff to:", json_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, help="Prompt for first image generation")
    parser.add_argument("--out_dir", default="outputs/step1_3")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
