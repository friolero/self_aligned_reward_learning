# self_aligned_reward_learning

This is the implementation for [Learning Reward for Robot Skills Using Large Language Models via Self-Alignment](https://openreview.net/pdf?id=Z19JQ6WFtJ). 


## Dependencies

The experiments were conducted on Ubuntu20.04:

* torch==1.13.1
* torchvision==0.14.1
* gymnasium==0.23.1
* wandb==0.12.21
* imageio==2.31.1
* moviepy==1.0.3
* stable_baselines=2.2.1
* APReL
* mani_skill2
* tulip 


## Launch self-aligned reward learning with LLM

Run the following command and input configurable arguments after each request is prompted and available options are listed (if select from a list): 

```bash
python3 launcher.py
```

Arguments:

> (1) skill_idx from [0-9] as listed;
>
> (2) reward option: 0 - oracle; 1 - llm_fixed; 2 - llm_updated, 3 - llm_fixed_last;
>
> (3) iteration steps: int (default: 100000 for sac, 1000 for ppo);
>
> (4) total steps: int; 
>
> (5) seeds: List[int].

