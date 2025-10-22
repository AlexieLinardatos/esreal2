import argparse, os, json
from typing import List, Tuple, Dict
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ---------- simple label set to query ----------
COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
        "traffic light","fire hydrant","stop sign","bench","bird","cat","dog","horse",
        "sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
        "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
        "baseball bat","baseball glove","skateboard","surfboard","tennis racket",
        "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
        "chair","couch","potted plant","bed","dining table","toilet","tv","laptop",
        "mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
        "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
EXTRA = ["man","woman","boy","girl","baby","tree","mountain","river","lake","sky","cloud",
         "street","building","bag","glasses"]
VOCAB = COCO + EXTRA
STOP = set("a an the of in on at to for with from by and or as is are was were be been being this that these those".split())

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def extract_phrases(caption: str) -> List[str]:
    cap = caption.lower()
    phrases = [m for m in VOCAB if " " in m and m in cap]
    tokens = [t.strip(".,!?;:()[]{}\"'") for t in cap.split() if t.strip()]
    tokens = [t for t in tokens if t not in STOP and len(t) > 2]
    tset = set(tokens)
    for s in VOCAB:
        if " " not in s and s in tset:
            phrases.append(s)
    # de-dup while keeping order
    seen, out = set(), []
    for p in phrases:
        if p not in seen:
            out.append(p); seen.add(p)
    if not out:
        out = tokens[:3]
    return out[:12]

def caption_image(img: Image.Image, device: str) -> str:
    from transformers import pipeline
    dev = 0 if device.startswith("cuda") else -1
    cap = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=dev)
    return cap(img)[0]["generated_text"]

def regenerate_image(prompt: str, device: str) -> Image.Image:
    from diffusers import AutoPipelineForText2Image
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=dtype)
    pipe.to(device)
    # Turbo-ish settings: tiny steps, no guidance
    return pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0.0).images[0]

def detect_owlvit(img: Image.Image, phrases: List[str], device: str, thr: float=0.15) -> Dict[str, List[Tuple[List[float], float]]]:
    if not phrases: return {}
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    proc = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    # OWL-ViT expects list-of-list for text queries
    inputs = proc(text=[phrases], images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    target_sizes = torch.tensor([img.size[::-1]], device=device)  # (h,w)
    res = proc.post_process_object_detection(out, target_sizes=target_sizes)[0]
    boxes = res.get("boxes", torch.zeros((0,4), device=device)).detach().cpu().numpy()
    scores = res.get("scores", torch.zeros((0,), device=device)).detach().cpu().numpy()
    labels = res.get("labels", torch.zeros((0,), device=device, dtype=torch.long)).detach().cpu().numpy()
    det = {p: [] for p in phrases}
    for b, s, li in zip(boxes, scores, labels):
        if s >= thr and 0 <= li < len(phrases):
            det[phrases[li]].append((b.tolist(), float(s)))
    return det

def annotate(img: Image.Image, det: Dict[str, List[Tuple[List[float], float]]]) -> Image.Image:
    im = img.copy().convert("RGB"); d = ImageDraw.Draw(im)
    try: font = ImageFont.load_default()
    except: font = None
    for label, items in det.items():
        for box, score in items:
            x0,y0,x1,y1 = box
            d.rectangle([x0,y0,x1,y1], outline=(255,0,0), width=3)
            txt = f"{label}:{score:.2f}"
            w = int(d.textlength(txt, font=font)) if hasattr(d,"textlength") else 7*len(txt)
            d.rectangle([x0, max(0,y0-14), x0+w+6, y0], fill=(255,0,0))
            d.text((x0+3, max(0,y0-13)), txt, fill=(255,255,255), font=font)
    return im

def clip_sim(img_a: Image.Image, img_b: Image.Image, device: str) -> float:
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = proc(images=[img_a, img_b], return_tensors="pt")
    pv = inputs["pixel_values"].to(device)
    with torch.no_grad():
        fa, fb = model.get_image_features(pixel_values=pv)
    fa = fa / fa.norm(dim=-1, keepdim=True)
    fb = fb / fb.norm(dim=-1, keepdim=True)
    return float((fa @ fb.T)[0,1].item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to input image")
    ap.add_argument("--out", default="outputs/quick_esreal_demo")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--skip_detect", action="store_true")
    args = ap.parse_args()

    device = "cuda" if (args.device=="auto" and torch.cuda.is_available()) else ("cpu" if args.device=="cpu" else args.device)
    ensure_dir(args.out)

    img = Image.open(args.image).convert("RGB")

    print(">> captioning…")
    caption = caption_image(img, device)
    with open(os.path.join(args.out,"caption.txt"),"w",encoding="utf-8") as f: f.write(caption)
    print("caption:", caption)

    print(">> regenerating from caption…")
    regen = regenerate_image(caption, device)
    regen.save(os.path.join(args.out,"regen.png"))

    det = {}
    annotated = img
    if not args.skip_detect:
        print(">> extracting phrases + OWL-ViT detection…")
        phrases = extract_phrases(caption)
        det = detect_owlvit(img, phrases, device, thr=0.15)
        annotated = annotate(img, det)
        annotated.save(os.path.join(args.out,"annotated.png"))

    print(">> CLIP image-image similarity…")
    sim = clip_sim(img, regen, device)

    rows = []
    for k, items in det.items():
        rows.append({"label": k, "present": bool(items), "n_boxes": len(items), "max_score": round(max([s for _,s in items], default=0.0),3)})
    report = {"caption": caption, "clip_image_similarity": round(sim,4), "detections": rows}
    with open(os.path.join(args.out,"report.json"),"w",encoding="utf-8") as f: json.dump(report, f, indent=2)

    with open(os.path.join(args.out,"report.md"),"w",encoding="utf-8") as f:
        f.write("# ESREAL-style quick demo\n\n")
        f.write(f"**Caption:** {caption}\n\n")
        f.write(f"**CLIP similarity (orig vs regen):** {sim:.4f}\n\n")
        if rows:
            f.write("| label | present | n_boxes | max_score |\n|---|---:|---:|---:|\n")
            for r in rows: f.write(f"| {r['label']} | {r['present']} | {r['n_boxes']} | {r['max_score']:.3f} |\n")

    print("DONE. Files in:", args.out)

if __name__ == "__main__":
    main()
