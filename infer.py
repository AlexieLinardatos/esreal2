
import os
import gc
import json
from typing import Optional
import pandas as pd

import fire
import torch

from PIL import Image
from tqdm import tqdm

from accelerate import Accelerator

from lavis.models import load_model_and_preprocess
from torch.utils.data import DataLoader, Subset, Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class EsrealImageDataset(Dataset):
    def __init__(self, df_path, image_dir):
        self.df = pd.read_csv(df_path)
        self.image_dir = image_dir  # kept for future use if you ever need to join paths

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "image_id": idx,
            "image_path": row["image_path"],  # path in CSV should already be absolute/relative
        }

    def __len__(self):
        return len(self.df)


def main(
    target_checkpoint: Optional[str],
    save_dir: str,
    save_filename: str,
    start_index: int,
    interval: int,
    df_path: str,
    image_dir: str,  # kept for signature parity (not used directly here)
    prompt: str,
    batch_size: int,
    num_workers: int,
):
    # create accelerator
    accelerator = Accelerator()
    print("accelerate device:", accelerator.device)
    print("cuda available:", torch.cuda.is_available())

    # load model
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5_instruct",
        model_type="flant5xl",
        is_eval=True,
    )

    if target_checkpoint is not None and str(target_checkpoint).lower() != "none":
        model.load_state_dict(torch.load(target_checkpoint, map_location="cpu"), strict=False)

    # tokenizer
    t5_tokenizer = model.t5_tokenizer

    # dataset / loader
    full_dataset = EsrealImageDataset(df_path, image_dir)
    max_size = len(full_dataset)
    test_dataset = Subset(full_dataset, range(start_index, min(start_index + interval, max_size)))
    # On Windows, num_workers=0 avoids multiprocessing issues
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # prepare accelerator
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    os.makedirs(save_dir, exist_ok=True)

    # Final output path (single chunk for this slice)
    chunk_name = save_filename.replace(".jsonl", f"__{start_index}__{min(start_index + interval, max_size)}.jsonl")
    save_path = os.path.join(save_dir, chunk_name)

    # If resuming, you may choose to remove existing file or keep appending.
    # Here we start fresh for this slice:
    if accelerator.is_main_process and os.path.exists(save_path):
        os.remove(save_path)

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            # handle list vs tensor ids robustly
            image_ids = batch["image_id"]
            if hasattr(image_ids, "tolist"):
                image_ids = image_ids.tolist()

            image_paths = batch["image_path"]

            # load & preprocess images
            proc_images = []
            for p in image_paths:
                with Image.open(p).convert("RGB") as im:
                    proc_images.append(vis_processors["eval"](im))
            images = torch.stack(proc_images).to(accelerator.device)

            # generate (keep it shorter/faster if you like with max_length)
            outputs = accelerator.unwrap_model(model).generate(
                samples={"image": images, "prompt": [prompt] * images.shape[0]},
                top_p=1.0,
                # max_length=32,  # uncomment to speed up / shorten outputs
                # temperature=0.7,
            )

            # safe decode (LAVIS may already return strings)
            if isinstance(outputs[0], str):
                output_texts = outputs
            else:
                output_texts = t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # stream-append this batch to disk
            if accelerator.is_main_process:
                with open(save_path, "a", encoding="utf-8") as f:
                    for i, t in zip(image_ids, output_texts):
                        json.dump({"image_id": int(i), "caption": t}, f, ensure_ascii=False)
                        f.write("\n")

            torch.cuda.empty_cache()
            gc.collect()

    if accelerator.is_main_process:
        print(f"✅ Wrote captions to: {save_path}")


if __name__ == "__main__":
    fire.Fire(main)






# import os
# import gc
# import json
# from typing import Optional
# import pandas as pd

# import fire
# import torch
# import torch.distributed as dist

# from PIL import Image
# from tqdm import tqdm

# from accelerate import Accelerator

# from lavis.datasets.builders import load_dataset
# from lavis.models import load_model_and_preprocess
# from lavis.models import model_zoo
# from torch.utils.data import DataLoader, Subset, Dataset

# os.environ["TOKENIZERS_PARALLELISM"] = "true"


# class EsrealImageDataset(Dataset):
#     def __init__(self, df_path, image_dir):
#         self.df = pd.read_csv(df_path)
#         self.image_dir = image_dir

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         return {
#             "image_id": idx,
#             "image_path": row["image_path"],
#         }

#     def __len__(self):
#         return len(self.df)

# def main(
#     target_checkpoint: Optional[str],
#     save_dir: str,
#     save_filename: str,
#     start_index: int,
#     interval: int,
#    # dataset_name: str,
#     df_path: str,
#     image_dir: str,
#     prompt: str,
#     batch_size: int,
#     num_workers: int,
# ):
#     # create accelerator
#     accelerator = Accelerator()

#     # load model
#     model, vis_processors, _ = load_model_and_preprocess(
#         name="blip2_t5_instruct", 
#         #name="blip2_t5_instruct_lora_with_value_head",
#         model_type="flant5xl",
#         is_eval=True,
#     )

#     if target_checkpoint is not None:
#         model.load_state_dict(torch.load(target_checkpoint, map_location="cpu"), strict=False)

#     # load tokenizer
#     t5_tokenizer = model.t5_tokenizer
#     tokenized = t5_tokenizer(prompt, return_tensors="pt")

#     # load dataset
#     full_dataset = EsrealImageDataset(df_path, image_dir)
#     max_size = len(full_dataset)
#     test_dataset = Subset(full_dataset, range(start_index, min(start_index + interval, max_size)))
#     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

#     # prepare accelerator
#     model, test_dataloader = accelerator.prepare(model, test_dataloader)

#     # inference
#     inference_results = []

#     with torch.no_grad():
#         for batch in tqdm(test_dataloader):
#             image_ids = batch["image_id"].tolist()
#             image_paths = batch["image_path"]
#             images = [vis_processors["eval"](Image.open(image_path).convert("RGB")) for image_path in image_paths]
#             images = torch.stack(images).to(accelerator.device)
#             input_ids = tokenized.input_ids.repeat_interleave(images.shape[0], dim=0).to(accelerator.device)
#             outputs = accelerator.unwrap_model(model).generate(
#                 samples={"image": images, "prompt": [prompt] * images.shape[0]},
#                 top_p=1.0,
#             )
#             if isinstance(outputs[0], str):
#                 output_texts = outputs  # Already decoded
#             else:
#                 output_texts = t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)
#             inference_results.extend(
#                 [
#                     {"image_id": image_id, "caption": output_text}
#                     for image_id, output_text in zip(image_ids, output_texts)
#                 ]
#             )

#             torch.cuda.empty_cache()
#             gc.collect()

#     # gather results
#     if accelerator.is_main_process:
#         gathered_results = [None] * dist.get_world_size()
#         dist.gather_object(inference_results, gathered_results)
#     else:
#         dist.gather_object(inference_results)

#     # save results
#     if accelerator.is_main_process:
#         # reformat gathered_results
#         final_results = []
#         for res in gathered_results:
#             final_results.extend(res)
#         if interval < BIG_NUMBER:
#             save_filename = save_filename.replace(".jsonl", f"__{start_index}__{start_index + interval}.jsonl")
#         save_path = os.path.join(save_dir, save_filename)
#         with open(save_path, "w") as f:
#             for item in final_results:
#                 json.dump(item, f, ensure_ascii=False)
#                 f.write("\n")


# if __name__ == "__main__":
#     fire.Fire(main)
