import os
import sys
import argparse
import functools
import time
from copy import deepcopy
from typing import List

import torch
import numpy as np

from lavis.datasets.builders import load_dataset
from lavis.models import load_model_and_preprocess

import trlx
from trlx.data.default_configs import (
    TRLConfig,
    TrainConfig,
    TokenizerConfig,
    OptimizerConfig,
    SchedulerConfig,
    PPOConfig,
)
from trlx.data.configs import NNModelConfig

# For converting tensors -> PIL for the reward pipeline
from torchvision import transforms as T

# Make sure we can import reward_model.* if this lives in GroundingDINO/
sys.path.append(os.path.dirname(__file__))

from reward_model.registry import registry  # your local ESREAL reward registry

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Function '{func.__name__}' took: {end_time - start_time:.4f} seconds")
        return result

    return wrapper


def default_ppo_config(model, model_copy, args):
    return TRLConfig(
        train=TrainConfig(
            seq_length=args.seq_length,
            epochs=100,
            total_steps=10000,
            batch_size=args.batch_size,
            checkpoint_interval=100,
            eval_interval=100,
            pipeline="VLMDatasetPipeline",
            trainer="AccelerateVLMPPOTrainer",
            checkpoint_dir=args.checkpoint_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
        ),
        model=NNModelConfig(
            model_arch_type="seq2seq",
            ref_to_model=model,
            model_copy=model_copy,
            model_init_kwargs=dict(),
        ),
        tokenizer=TokenizerConfig(
            tokenizer_path=args.model_path,
            padding_side="left",
            truncation_side="right",
        ),
        optimizer=OptimizerConfig(
            name="adamw",
            kwargs=dict(
                lr=args.lr,
                betas=(0.9, 0.95),
                eps=1.0e-8,
                weight_decay=1.0e-6,
            ),
        ),
        scheduler=SchedulerConfig(
            name="cosine_annealing",
            kwargs=dict(T_max=1e12, eta_min=3e-5),
        ),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=128,
            chunk_size=args.chunk_size,
            ppo_epochs=4,
            init_kl_coef=args.init_kl_coef,
            target=None,
            horizon=10000,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=1,
            scale_reward="ignored",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
        ),
    )


def create_reference_model(model):
    parameter_names = [n for n, _ in model.named_parameters()]
    ref_model = deepcopy(model)
    for param_name in parameter_names:
        param = ref_model.get_parameter(param_name)
        param.requires_grad = False
    return ref_model.eval()


def main(args):
    dataset_name = args.dataset_name
    alpha = args.alpha
    is_rec_penalty = args.is_rec_penalty

    # device index for LAVIS model (-1 = CPU)
    device = int(os.environ.get("LOCAL_RANK", 0)) if torch.cuda.is_available() else -1

    # -------------------------------
    # 1) Load BLIP2 + value head
    # -------------------------------
    model, _, _ = load_model_and_preprocess(
        name="blip2_t5_instruct_lora_with_value_head",
        model_type="flant5xl",
        is_eval=False,
        device=device,
    )
    model_copy = create_reference_model(model)

    config = default_ppo_config(model, model_copy, args)

    # -------------------------------
    # 2) Load dataset
    # -------------------------------
    dataset = load_dataset(dataset_name, df_path=args.df_path, image_dir=args.image_dir)

    # -------------------------------
    # 3) Initialize local reward pipeline
    # -------------------------------
    # registry.device should be "cuda" or "cpu"
    if torch.cuda.is_available() and device != -1:
        registry.device = "cuda"
    else:
        registry.device = "cpu"
    print(f"[ESREAL] Reward registry using device: {registry.device}")
    registry.initialize()  # loads GDINO, SDXL, CLIP, RewardCalculator, etc.

    to_pil = T.ToPILImage()

    @timeit
    def dense_reward_fn(
        images: torch.Tensor,
        samples: List[str],
        prompts: List[str],
        outputs: List[str],
        tokenizer,
        **kwargs,
    ) -> List[float]:
        """
        This is called by TRLX during PPO.
        We:
          - convert images -> PIL
          - tokenize captions very simply (split on spaces)
          - call registry.reward_pipeline (your local ESREAL reward)
          - build per-token dense rewards, same formula as original script.
        """

        # images: (B, C, H, W) on GPU
        B = images.shape[0]

        # Convert to list[PIL.Image] on CPU
        pil_images = [
            to_pil(images[i].detach().cpu()).convert("RGB") for i in range(B)
        ]

        # Captions as plain strings
        captions = [str(o) for o in outputs]

        # Very simple tokenization (same as reward_driver.py / run_reward_local.py)
        tokenized_outputs = [c.split() for c in captions]

        # Call your local ESREAL reward pipeline
        (
            mean_rec_reward,
            mean_obj_penalty,
            mean_att_penalty,
            mean_rel_penalty,
            mean_pos_penalty,
        ) = registry.reward_pipeline(
            pil_images,       # List[Image.Image]
            captions,         # List[str]
            tokenized_outputs # List[List[str]]
        )

        dense_rewards = []

        for i in range(B):
            L = len(tokenized_outputs[i])
            dense_reward = []

            for j in range(L):
                if is_rec_penalty:
                    dense_reward.append(
                        (
                            (0 if mean_rec_reward[i][j] == 0 else alpha * (mean_rec_reward[i][j] - 1))
                            + mean_obj_penalty[i][j]
                            + mean_att_penalty[i][j]
                            + mean_rel_penalty[i][j]
                            + mean_pos_penalty[i][j]
                        )
                    )
                else:
                    dense_reward.append(
                        (
                            alpha * mean_rec_reward[i][j]
                            + mean_obj_penalty[i][j]
                            + mean_att_penalty[i][j]
                            + mean_pos_penalty[i][j]
                        )
                    )

            dense_rewards.append(dense_reward)

        reward_dict = {
            "dense_rewards": dense_rewards,
            "rec_reward": mean_rec_reward,
            "obj_penalty": mean_obj_penalty,
            "att_penalty": mean_att_penalty,
            "rel_penalty": mean_rel_penalty,
            "pos_penalty": mean_pos_penalty,
        }

        return dense_rewards, reward_dict

    # -------------------------------
    # 4) PPO training with TRLX
    # -------------------------------
    trlx.train(
        reward_fn=dense_reward_fn,
        prompts=dataset["train"],
        eval_prompts=dataset["val"],
        config=config,
        args=args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # NOTE: triton_server_url removed – we are fully local now
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--df_path", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--task_name", type=str, choices=["short_caption", "long_caption", "vqa"])
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--chunk_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seq_length", type=int)
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--repetition_penalty", type=float)
    parser.add_argument("--init_kl_coef", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--is_rec_penalty", type=bool)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--resume_from_checkpoint", type=str)
    args = parser.parse_args()

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.experiment_name)

    main(args)
