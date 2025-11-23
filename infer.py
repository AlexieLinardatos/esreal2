# at top (new imports)
import os, json, signal, sys
from typing import Optional, Set
import fire
# ... keep your existing imports ...

def _load_state(state_path: Optional[str]) -> dict:
    if not state_path or not os.path.exists(state_path):
        return {"next_index": None, "done_ids": []}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"next_index": None, "done_ids": []}

def _save_state(state_path: Optional[str], state: dict) -> None:
    if not state_path:
        return
    tmp = state_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, state_path)

def _append_jsonl(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

def _determine_range(n_rows: int, start_index: int, interval: int) -> range:
    if interval and interval > 0:
        end = min(n_rows, start_index + interval)
    else:
        end = n_rows
    return range(start_index, end)

def main(
    target_checkpoint: Optional[str],
    save_dir: str,
    save_filename: str,
    start_index: int,
    interval: int,
    df_path: str,
    image_dir: str,
    state_path: Optional[str] = None,
    save_every: int = 50,
):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, save_filename)

    dataset = EsrealImageDataset(df_path=df_path, image_dir=image_dir)
    n_rows = len(dataset)

    # ---- load state (resume) ----
    state = _load_state(state_path)
    done_ids: Set[int] = set(state.get("done_ids", []))

    if state.get("next_index") is not None:
        start_index = max(start_index, int(state["next_index"]))

    work_range = _determine_range(n_rows, start_index, interval)

    # ---- graceful shutdown hooks ----
    stopped = {"flag": False}
    def _handle_stop(signum, frame):
        stopped["flag"] = True
        print(f"\n[signal {signum}] received -> will save state and exit...", file=sys.stderr)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_stop)
        except Exception:
            pass  # some platforms may not allow setting handlers

    # ---- load your model(s) (example with LAVIS; adjust as needed) ----
    accelerator = Accelerator()
    device = accelerator.device
    # Example:
    # model, vis_processors, txt_processors = load_model_and_preprocess(
    #     name="blip_caption", model_type="base_coco", is_eval=True, device=device
    # )

    # ---- iterate with periodic saves ----
    processed_since_save = 0
    last_index = start_index

    try:
        for idx in tqdm(work_range, desc="ESREAL Inference"):
            if idx in done_ids:
                last_index = idx + 1
                continue

            sample = dataset[idx]
            img_path = sample["image_path"]

            # --- YOUR INFERENCE HERE ---
            # image = Image.open(img_path).convert("RGB")
            # result = model.generate(...)  # placeholder
            result = {"image_id": idx, "image_path": img_path, "pred": "TODO"}  # stub

            _append_jsonl(out_path, result)
            done_ids.add(idx)
            processed_since_save += 1
            last_index = idx + 1

            # periodic state flush
            if processed_since_save >= max(1, save_every):
                _save_state(state_path, {"next_index": last_index, "done_ids": sorted(done_ids)})
                processed_since_save = 0

            if stopped["flag"]:
                break

    except KeyboardInterrupt:
        print("\n[KeyboardInterrupt] stopping...", file=sys.stderr)

    finally:
        # final save
        _save_state(state_path, {"next_index": last_index, "done_ids": sorted(done_ids)})
        print(f"State saved to {state_path}. Next index: {last_index}")

if __name__ == "__main__":
    fire.Fire(main)


    
