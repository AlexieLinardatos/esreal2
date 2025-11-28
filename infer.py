import os
import gc
import json
import signal
import sys
from typing import Optional, Set

import fire
import torch

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator

from lavis.datasets.builders import load_dataset
from lavis.models import load_model_and_preprocess

os.environ["TOKENIZERS_PARALLELISM"] = "true"


# ---------------------------
# State & IO helpers
# ---------------------------

def _load_state(state_path: Optional[str]) -> dict:
    """
    Load resume state from JSON, or return a default if missing/corrupt.
    State format:
        {
            "next_index": Optional[int],  # currently not strictly needed
            "done_ids": List[int]         # image_ids already processed
        }
    """
    if not state_path or not os.path.exists(state_path):
        return {"next_index": None, "done_ids": []}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"next_index": None, "done_ids": []}


def _save_state(state_path: Optional[str], state: dict) -> None:
    """
    Atomically write state to disk (via .tmp + os.replace).
    """
    if not state_path:
        return
    tmp = state_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, state_path)


def _append_jsonl(path: str, obj: dict) -> None:
    """
    Append a single JSON object as one line to a JSONL file.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------------------
# Main
# ---------------------------

def main(
    target_checkpoint: Optional[str],
    save_dir: str,
    save_filename: str,
    start_index: int,
    interval: int,
    dataset_name: str,
    df_path: str,
    image_dir: str,
    prompt: str,
    batch_size: int,
    num_workers: int,
    state_path: Optional[str] = None,
    save_every: int = 50,
):
    """
    Runs image->caption inference over a chunk of the test dataset, with:
      - start_index + interval chunking (like the original script),
      - JSONL streaming output,
      - resume support via state_path,
      - graceful shutdown on Ctrl+C / SIGTERM.

    Each output line is:
        {"image_id": int, "caption": str}
    """

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, save_filename)

    # ---------------------------
    # Load resume state
    # ---------------------------
    state = _load_state(state_path)
    done_ids: Set[int] = set(state.get("done_ids", []))

    # We keep start_index/interval semantics the same as the original script:
    #   - The chunk is [start_index, start_index + interval)
    #   - Resume is handled by skipping image_ids that are already in done_ids.
    # This avoids having to re-slice the dataset in a complicated way.
    # (next_index is kept in state but not strictly required.)

    # ---------------------------
    # Signal handling (Ctrl+C etc.)
    # ---------------------------
    stopped = {"flag": False}

    def _handle_stop(signum, frame):
        stopped["flag"] = True
        print(f"\n[signal {signum}] received -> will save state and exit...", file=sys.stderr)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_stop)
        except Exception:
            # Some platforms (e.g., Jupyter) may not allow setting handlers
            pass

    # ---------------------------
    # Accelerator + model
    # ---------------------------
    accelerator = Accelerator()

    # Load model exactly as in the original script
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5_instruct_lora_with_value_head",
        model_type="flant5xl",
        is_eval=True,
    )

    if target_checkpoint is not None:
        accelerator.print(f"Loading checkpoint from {target_checkpoint}")
        ckpt = torch.load(target_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)

    # Load tokenizer from model
    t5_tokenizer = model.t5_tokenizer
    tokenized = t5_tokenizer(prompt, return_tensors="pt")

    # ---------------------------
    # Dataset + chunking
    # ---------------------------
    dataset = load_dataset(dataset_name, df_path=df_path, image_dir=image_dir, include_pil=False)
    test_dataset = dataset["test"]
    max_size = len(test_dataset)
    accelerator.print(f"Total test dataset size: {max_size}")

    # Define the chunk [start_index, min(start_index + interval, max_size))
    end_index = min(start_index + interval, max_size)
    if start_index >= max_size:
        accelerator.print(f"start_index ({start_index}) >= dataset size ({max_size}). Nothing to do.")
        return

    chunk_indices = range(start_index, end_index)
    test_subset = Subset(test_dataset, chunk_indices)
    test_dataloader = DataLoader(
        test_subset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    # Prepare with accelerator
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    # ---------------------------
    # Inference loop
    # ---------------------------
    processed_since_save = 0

    try:
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="ESREAL Inference"):
                if stopped["flag"]:
                    break

                image_ids = batch["image_id"].tolist()
                image_paths = batch["image_path"]

                # Skip already-done ids (for resume)
                keep_mask = [img_id not in done_ids for img_id in image_ids]
                if not any(keep_mask):
                    # Entire batch already processed
                    continue

                # Filter the batch to only new samples
                kept_ids = [img_id for img_id, keep in zip(image_ids, keep_mask) if keep]
                kept_paths = [p for p, keep in zip(image_paths, keep_mask) if keep]

                # Load & preprocess images
                images = [vis_processors["eval"](Image.open(p).convert("RGB")) for p in kept_paths]
                images = torch.stack(images).to(accelerator.device)

                # Repeat the tokenized prompt to match batch size
                input_ids = tokenized.input_ids.repeat_interleave(images.shape[0], dim=0).to(accelerator.device)

                # Generate with underlying model
                outputs = accelerator.unwrap_model(model).generate(
                    images,
                    input_ids,
                    top_p=1.0,
                )
                output_texts = t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)

                # Write per-sample JSONL lines + update state
                for img_id, caption in zip(kept_ids, output_texts):
                    result = {
                        "image_id": int(img_id),
                        "image_path": img_path,
                        "caption": caption,
                    }
                    _append_jsonl(out_path, result)
                    done_ids.add(int(img_id))
                    processed_since_save += 1

                # free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # Periodic state save
                if processed_since_save >= max(1, save_every):
                    # We don't strictly need next_index; we keep it for compatibility.
                    # Here we set it to the max processed image_id + 1 (best-effort).
                    next_index = max(done_ids) + 1 if done_ids else None
                    _save_state(
                        state_path,
                        {"next_index": next_index, "done_ids": sorted(done_ids)},
                    )
                    processed_since_save = 0

    except KeyboardInterrupt:
        accelerator.print("\n[KeyboardInterrupt] stopping...", file=sys.stderr)

    finally:
        # Final state save
        next_index = max(done_ids) + 1 if done_ids else None
        _save_state(
            state_path,
            {"next_index": next_index, "done_ids": sorted(done_ids)},
        )
        accelerator.print(f"State saved to {state_path}. Processed {len(done_ids)} unique image_ids.")


if __name__ == "__main__":
    fire.Fire(main)
