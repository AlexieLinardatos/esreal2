# testscript.py
from lavis.models import load_model_and_preprocess
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5xl",
    is_eval=True,
    device=device
)

image = Image.new("RGB", (512, 512), color="gray")
image_tensor = vis_processors["eval"](image).unsqueeze(0).to(device)

samples = {"image": image_tensor, "prompt": "Describe this image."}
output = model.generate(samples)
print("🖼️ Caption:", output)
