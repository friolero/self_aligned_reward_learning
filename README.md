# self_aligned_reward_learning

This is the implementation for [Learning Reward for Robot Skills Using Large Language Models via Self-Alignment](https://arxiv.org/pdf/2405.07162).


## Dependencies

Install the dependencies for stable_baselines3

```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

Install the Python dependencies. 

```bash
cd self_aligned_reward_learning
pip install -r requirements.txt
cd ManiSkill2
pip install .
cd ../APReL
pip install .
```


## Launch self-aligned reward learning with LLM

1) Download ManiSkill2 asset data from [this link](https://drive.google.com/file/d/1wv_vAwjWlGZ-xaHnZDWAKH9aH2VphtjH/view?usp=sharing) and extract under `reward_learning` folder as `data`.

Alternatively you can download it by running:
```bash
python -m mani_skill.utils.download_asset partnet_mobility_cabine
mv data reward_learning/
```

2) Run the following command and input configurable arguments after each request is prompted and available options are listed (if select from a list): 

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


## Reproduce Paper Results

The learnt reward history are saved under `reward_learning/learnt_reward` for 10 tasks in figure 5 and 6. To train with history, pass corresponding history file according to the name to the argument `--learnt_reward_fn`. 

For example:

```bash
python3 main.py --skill 'push a swivel chair to a target 2D location on the ground' --seed 385 --reward_option llm_updated --iter_steps 100000 --total_steps 8000000  --learnt_reward_fn learnt_rewards/push_chair.pkl
```

## BibTex

You can cite this work at 
```
@article{zeng2024learning,
  title={Learning Reward for Robot Skills Using Large Language Models via Self-Alignment},
  author={Zeng, Yuwei and Mu, Yao and Shao, Lin},
  journal={arXiv preprint arXiv:2405.07162},
  year={2024}
}
```

Meanwhile, this repo is built upon APReL and ManiSkill2 with BibTex as follows:
```
@inproceedings{biyik2022aprel,
  title={Aprel: A library for active preference-based reward learning algorithms},
  author={B{\i}y{\i}k, Erdem and Talati, Aditi and Sadigh, Dorsa},
  booktitle={2022 17th ACM/IEEE International Conference on Human-Robot Interaction (HRI)},
  pages={613--617},
  year={2022},
  organization={IEEE}
}
@article{gu2023maniskill2,
  title={Maniskill2: A unified benchmark for generalizable manipulation skills},
  author={Gu, Jiayuan and Xiang, Fanbo and Li, Xuanlin and Ling, Zhan and Liu, Xiqiang and Mu, Tongzhou and Tang, Yihe and Tao, Stone and Wei, Xinyue and Yao, Yunchao and others},
  journal={arXiv preprint arXiv:2302.04659},
  year={2023}
}
```
