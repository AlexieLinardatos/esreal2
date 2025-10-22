from reward_model import registry

def run_reward(images, prompts, tokenized_prompts):
    registry.device = "cuda:0"
    registry.initialize()
    return registry.reward_pipeline(images, prompts, tokenized_prompts)
