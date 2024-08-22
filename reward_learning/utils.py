import ast
import glob
import importlib
import os
import pickle as pkl
import random
import sys
import time
from dataclasses import asdict

import numpy as np

import cv2
import gym
import openai
import torch
from logger import Logger
from mani_skill2_wrapper import (
    ManiSkill2DualArmMobileBaseTaskWrapper,
    ManiSkill2FixedBaseTaskWrapper,
    ManiSkill2MobileBaseTaskWrapper,
)

openai.api_key = os.getenv("OPENAI_API_KEY")


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def fix_softlink(data_dir: str):
    idx = 0
    while True:
        obs_1_fn = os.path.abspath(f"{data_dir}/{idx:09d}_obs_1.pkl")
        next_obs_0_fn = os.path.abspath(f"{data_dir}/{idx+1:09d}_obs_0.pkl")
        next_obs_1_fn = os.path.abspath(f"{data_dir}/{idx+1:09d}_obs_1.pkl")
        if not os.path.isfile(next_obs_1_fn):
            print(f"Finished for {idx} step files")
            break
        if (not os.path.isfile(next_obs_0_fn)) and (
            os.path.isfile(next_obs_1_fn)
        ):
            os.symlink(obs_1_fn, next_obs_0_fn)
            for view in ["wrist", "front"]:
                for mode in ["rgb", "depth"]:
                    img_1_fn = obs_1_fn.replace(
                        "obs_1.pkl", f"{view}_{mode}_1.png"
                    )
                    next_img_0_fn = next_obs_0_fn.replace(
                        "obs_0.pkl", f"{view}_{mode}_0.png"
                    )
                    if os.path.isfile(img_1_fn):
                        os.symlink(img_1_fn, next_img_0_fn)
        idx += 1


def skill2llm(skill: str, visual=False):
    answers = {
        "pick ycb object and transport to the target position": "0, 0, [1, 4, 5, 10, 12, 13, 14, 16, 17]",
        "pick cube and transport to the target position": "0, 0, [1, 4, 5, 10, 12, 13, 14, 16, 17]",
        "text2reward pick cube and transport to the target position": "0, 0, [1, 4, 5, 10, 12, 13, 14, 16, 17, 18]",
        "insert peg into the side hole": "0, 1, [1, 4, 5, 10, 12, 13, 14, 16, 17]",
        "open cabinet drawer": "0, 1, [1, 4, 5, 10, 12, 13, 14, 16, 17]",
        "text2reward open cabinet drawer": "0, 1, [1, 4, 5, 10, 12, 13, 14, 16, 17, 18]",
        "open cabinet door": "0, 1, [1, 4, 5, 10, 12, 13, 14, 16, 17]",
        "text2reward open cabinet door": "0, 1, [1, 4, 5, 10, 12, 13, 14, 16, 17, 18]",
        "push a swivel chair to a target 2D location on the ground": "0, 1, [1, 4, 5, 10, 12, 13, 14, 16, 17]",
        "text2reward push a swivel chair to a target 2D location on the ground": "0, 1, [1, 4, 5, 10, 12, 13, 14, 16, 17, 18]",
    }
    if visual:
        return answers[skill].replace("]", ", 6]")
    else:
        return answers[skill]


def skill2llm_hp_ranges(skill: str):
    """
    Initial one-time proposal query are done through GPT-4 web api.
    TODO: to use python api and auto parse init reward and hp ranges.
    """
    answers = {
        "pick cube and transport to the target position": {
            "approach_distance_weight": [0, np.inf],
            "grasp_reward_value": [0, np.inf],
            "goal_distance_weight": [0, np.inf],
            "collision_penalty_value": [-np.inf, 0],
            "maintaining_position_reward_value": [0, np.inf],
        },
        "text2reward pick cube and transport to the target position": {
            "weight_dist_gripper_cube": [0, np.inf],
            "weight_dist_cube_goal": [0, np.inf],
            "weight_grasping_cube": [0, np.inf],
            "weight_action_reg": [0, np.inf],
        },
        "pick ycb object and transport to the target position": {
            "approach_distance_weight": [0, np.inf],
            "grasp_reward_value": [0, np.inf],
            "goal_distance_weight": [0, np.inf],
            "collision_penalty_value": [-np.inf, 0],
            "maintaining_position_reward_value": [0, np.inf],
        },
        "insert peg into the side hole": {
            "distance_to_peg_weight": [0.0, np.inf],
            "alignment_ee_to_peg_weight": [0.0, np.inf],
            "distance_to_hole_weight": [0.0, np.inf],
            "alignment_peg_to_hole_weight": [0.0, np.inf],
            "grasp_reward_value": [0.0, np.inf],
            "collision_penalty_value": [-np.inf, 0.0],
            "force_threshold": [0.0, np.inf],
        },
        "open cabinet drawer": {
            "alignment_weight": [0, np.inf],
            "approach_weight": [0, np.inf],
            "grasp_weight": [0, np.inf],
            "pull_weight": [0, np.inf],
            "collision_penalty_weight": [0, np.inf],
            "non_progress_penalty_weight": [0, np.inf],
            "distance_penalty_weight": [0, np.inf],
        },
        "text2reward open cabinet drawer": {
            "w_distance": [0, np.inf],
            "w_goal": [0, np.inf],
            "w_action": [0, np.inf],
            "static_reward_value": [0, np.inf],
        },
        "open cabinet door": {
            "alignment_weight": [0, np.inf],
            "approach_weight": [0, np.inf],
            "grasp_weight": [0, np.inf],
            "pull_weight": [0, np.inf],
            "collision_penalty_weight": [0, np.inf],
            "non_progress_penalty_weight": [0, np.inf],
            "distance_penalty_weight": [0, np.inf],
        },
        "text2reward open cabinet door": {
            "w_distance": [0, np.inf],
            "w_goal": [0, np.inf],
            "w_action": [0, np.inf],
            "static_reward_value": [0, np.inf],
        },
        "push a swivel chair to a target 2D location on the ground": {
            "approach_weight": [0.1, np.inf],
            "movement_weight": [0.1, np.inf],
            "collision_penalty_weight": [0.1, np.inf],
        },
        "text2reward push a swivel chair to a target 2D location on the ground": {
            "w_dist": [0.0, np.inf],
            "w_target": [0.0, np.inf],
            "w_vel": [0.0, np.inf],
            "w_tilt": [0.0, np.inf],
            "w_reg": [0.0, np.inf],
        },
    }
    return answers[skill]


def llm_parse_action_obs_params(skill: str, use_llm_obs=True):
    idx2obs_name = {
        0: "original_obs",
        1: "joint_positions",
        2: "joint_velocities",
        3: "joint_reaction_forces",
        4: "ee_pose",
        5: "ee_reaction_force",
        6: "wrist_rgb",
        7: "wrist_depth",
        8: "front_rgb",
        9: "front_depth",
        10: "target_object_pose",
        11: "object_poses",
        12: "goal_pose",
        13: "distance_to_target",
        14: "distance_to_goal",
        15: "distance_to_init",
        16: "contacted",
        17: "collided",
        18: "action",
    }
    selected_obs_indices = [1, 4, 5, 10, 12, 13, 14, 16, 17]
    if "text2reward" in skill:
        selected_obs_indices.append(18)
    # if include 0 will only use environment original state for policy
    if not use_llm_obs:
        selected_obs_indices.insert(0, 0)
    relevant_obs_names = [idx2obs_name[idx] for idx in selected_obs_indices]
    return relevant_obs_names


def init_env(cfg):
    relevant_obs_names = llm_parse_action_obs_params(
        skill=cfg.skill, use_llm_obs=cfg.use_llm_obs
    )
    assert "maniskill2" in cfg.wandb_group, "Unsupported environment"
    if cfg.task_info.env_name in [
        "PickCube-v0",
        "PickSingleYCB-v0",
        "PegInsertionSide-v0",
    ]:
        maniskill2_wrapper = ManiSkill2FixedBaseTaskWrapper
    elif cfg.task_info.env_name in [
        "OpenCabinetDrawer-v1",
        "OpenCabinetDoor-v1",
    ]:
        maniskill2_wrapper = ManiSkill2MobileBaseTaskWrapper
    elif cfg.task_info.env_name == "PushChair-v1":
        maniskill2_wrapper = ManiSkill2DualArmMobileBaseTaskWrapper
    env = maniskill2_wrapper(
        env_name=cfg.task_info.env_name,
        target_object_name=cfg.task_info.target_object_name,
        target_pose_key=cfg.task_info.target_pose_key,
        goal_pose_key=cfg.task_info.goal_pose_key,
        distance_to_target_key=cfg.task_info.distance_to_target_key,
        distance_to_goal_key=cfg.task_info.distance_to_goal_key,
        distance_to_init_key=cfg.task_info.distance_to_init_key,
        max_steps=cfg.task_info.max_steps,
        control_mode=cfg.control_mode,
        reward_mode=cfg.reward_mode,
        relevant_obs_names=relevant_obs_names,
        ckpt_dir=f"ckpts/{cfg.wandb_group}/{cfg.wandb_run_name}",
    )
    return env


def init_logger(cfg, run_id=None):
    run_cfg = {}
    for k, v in asdict(cfg).items():
        if isinstance(v, dict):
            run_cfg.update(v)
        else:
            run_cfg[k] = v
    if run_id is not None:
        run_cfg["run_id"] = run_id
    else:
        run_cfg["run_id"] = None
    return Logger(run_cfg=run_cfg)


def create_relabel_reward_func(
    llm_reward_func, tmp_reward_func_fn="tmp_llm_reward_func.py"
):
    lines = [l for l in llm_reward_func.strip().split("\n") if len(l) > 0]
    assert lines[0].startswith("def "), "Expecting a function definition"
    new_func_lines = [lines[0].replace("obs,", "obs, reward_vars,")]
    for line in lines[1:]:
        if "print" in line:
            continue
        new_line = line.replace("\t", "    ")
        indentation = ""
        for c in line:
            if c == " ":
                indentation += c
            else:
                break
        if "=" in line:
            var_name = line.replace(indentation, "").split()[0]
            new_line = f"{indentation}try:\n    {line}\n{indentation}except:\n    {indentation}if '{var_name}' in reward_vars.keys():        {indentation}{var_name} = reward_vars['{var_name}']"
        # Local numerical variable parsing for reward features storage
        if "return " in line:
            return_str = (
                "vars = {}\n"
                + f"{indentation}for k, v in locals().items():\n{indentation}    if type(v) in [np.int, np.float32, np.float64, int, float, bool, np.bool_]:\n{indentation}        vars[k] = v\n{indentation}return"
            )
            new_line = new_line.replace("return", return_str)
            new_line += ", vars"
        new_func_lines.append(new_line)
    new_func_str = "\n".join(new_func_lines)
    with open(tmp_reward_func_fn, "w") as fp:
        fp.write(
            "import numpy as np\nfrom scipy.spatial.distance import cdist\n\n"
        )
        fp.write(new_func_str)

    tmp_llm_reward_func = importlib.import_module(
        tmp_reward_func_fn.replace(".py", "")
    )
    importlib.reload(tmp_llm_reward_func)
    from tmp_llm_reward_func import get_reward

    return get_reward


def parse_reward_func(llm_reward_func):
    hps = llm_reward_func[
        llm_reward_func.find("(") + 1 : llm_reward_func.find(")")
    ].split(", ")
    hps = {hp.split("=")[0]: eval(hp.split("=")[1]) for hp in hps if "=" in hp}
    return_var = llm_reward_func[
        llm_reward_func.find("return ") + len("return ") :
    ].strip()
    reward_eqs = [
        l for l in llm_reward_func.split("\n") if f"{return_var} = " in l
    ]
    reward_terms = []
    for eq in reward_eqs:
        reward_terms += eq.strip().split(" = ")[1].split(" + ")
    return hps, reward_terms, return_var


def update_llm_reward_func(llm_reward_func, updated_hps, init_llm_reward_func):

    init_def_line = llm_reward_func[
        init_llm_reward_func.find("def") : init_llm_reward_func.find("):")
        + len("):")
    ]
    updated_def_line = init_def_line[: init_def_line.find("obs") + len("obs")]
    for k, v in updated_hps.items():
        updated_def_line += f", {k}={v}"
    updated_def_line += "):"
    updated_llm_reward_func = init_llm_reward_func.replace(
        init_def_line, updated_def_line
    )
    return updated_llm_reward_func


def retrieve_in_memory_next_obs(replay_buffer, pos_indices):
    next_obses = []
    for idx in pos_indices:
        next_obs = {}
        for k in replay_buffer._obs:
            next_obs[k] = replay_buffer._obs[k][idx].numpy()
        next_obses.append(next_obs)
    return next_obses


def retrieve_in_memory_ep_rewards(
    replay_buffer,
    n_recent: int,
    last_step_reward: bool = True,
    sort: bool = False,
    start_high: bool = True,
):
    rewards = []
    indices = []
    n_data = min(n_recent, len(replay_buffer))

    ep_reward = 0
    ep_indices = []
    for idx in range(n_data):
        ep_reward += replay_buffer._rewards[idx].item()
        ep_indices.append(idx)
        if replay_buffer._dones[idx]:
            if last_step_reward:
                rewards.append(replay_buffer._rewards[idx].item())
                indices.append(idx)
            else:
                rewards.append(ep_reward)
                indices.append(ep_indices)
            ep_reward = 0
            ep_indices = []

    if sort:
        indices = [
            idx for _, idx in sorted(zip(rewards, indices), reverse=start_high)
        ]
        rewards = sorted(rewards, reverse=start_high)
    return rewards, indices


def partition_and_sample(rewards, n_bins=10):
    freq, bin_edges = np.histogram(
        rewards, bins=n_bins, range=[min(rewards), max(rewards)]
    )
    bins = {i: [] for i in range(n_bins)}
    for idx, reward in enumerate(rewards):
        for i, bin_edge in enumerate(bin_edges[:-1]):
            if (reward >= bin_edge) and (reward <= bin_edges[i + 1]):
                bins[i].append(idx)
                break
    n_sample = {}
    for k, v in bins.items():
        if len(v) > 0:
            n_sample[k] = 1
    sample_idx = []
    for k in n_sample:
        sample_idx.append(
            np.random.choice(bins[k], size=n_sample[k], replace=False)
        )
    sample_idx = np.concatenate(sample_idx, -1).tolist()
    if (len(bins[n_bins - 1]) > 0) and (0 not in sample_idx):
        sample_idx.append(0)
    return sample_idx


def sample_from_replay_hist(
    replay_source, n_recent, init_hps, reward_terms, return_var, n_bins=10
):
    raw_rewards, indices = retrieve_in_memory_ep_rewards(
        replay_source,
        n_recent,
        last_step_reward=True,
        sort=True,
        start_high=True,
    )
    sample_indices = partition_and_sample(raw_rewards, n_bins=n_bins)
    rewards = [raw_rewards[idx] for idx in sample_indices]
    sample_pos_indices = [indices[idx] for idx in sample_indices]
    next_obses = retrieve_in_memory_next_obs(replay_source, sample_pos_indices)

    obs_vars = []
    reward_vars = []
    for idx in sample_pos_indices:
        step_var = {k: v[idx] for k, v in replay_source._reward_vars.items()}
        obs_var, reward_var = parse_step_var(
            step_var, init_hps, reward_terms, return_var
        )
        obs_vars.append(obs_var)
        reward_vars.append(reward_var)
    return rewards, next_obses, init_hps, obs_vars, reward_vars


def parse_step_var(step_var, init_hps, reward_terms, return_var):
    redudant_key = [
        "success",
        "episodic_reward",
        "episode_length",
        "episode_dist2tgt",
        "episode_dist2goal",
        "views",
        "action",
        "reward",
        "done",
    ]

    obs_var = {}
    reward_var = {}
    for key in redudant_key + list(init_hps.keys()) + [return_var]:
        if key in step_var:
            step_var.pop(key)
    for key in step_var:
        if key in reward_terms:
            reward_var[key] = step_var[key]
        else:
            obs_var[key] = step_var[key]
    return obs_var, reward_var


def rank(list_to_rank):
    rank = sorted(
        range(len(list_to_rank)), key=list_to_rank.__getitem__, reverse=True
    )
    return rank


def normalize_hps(hps, hp_ranges):
    assert hps.keys() == hp_ranges.keys(), "Inconsistent hp and their ranges."
    normalized_hps = {}
    for k, v in hps.items():
        min_v, max_v = hp_ranges[k]
        normalized_hps[k] = (float(v) - min_v) / (max_v - min_v)
    return normalized_hps


def denormalize_hps(normalized_hps, hp_ranges):
    assert (
        normalized_hps.keys() == hp_ranges.keys()
    ), "Inconsistent hp and their ranges."
    hps = {}
    for k, v in normalized_hps.items():
        min_v, max_v = hp_ranges[k]
        hps[k] = (max_v - min_v) * v + min_v
    return hps


def llm(prompt, model="gpt-4-0613", temperature=1.0, chunk_size=4):
    for attempt in range(1000):
        try:
            response_cur = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": prompt}],
                temperature=temperature,
                n=chunk_size,
            )
            break
        except Exception as e:
            if attempt >= 10:
                chunk_size = max(int(chunk_size / 2), 1)
                print("Current Chunk Size", chunk_size)
                print(f"Attempt {attempt+1} failed with error: {e}")
                time.sleep(1)
    if response_cur is None:
        print("Code terminated due to too many failed attempts!")
        return None, None, None, None
    return (
        response_cur["choices"],
        response_cur["usage"]["prompt_tokens"],
        response_cur["usage"]["completion_tokens"],
        response_cur["usage"]["total_tokens"],
    )
