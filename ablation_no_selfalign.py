import argparse
import glob
import importlib
import os
import pickle as pkl
import types
from copy import deepcopy
from functools import partial
from typing import Callable, Union

import gymnasium as gym
import numpy as np
import wandb
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from wandb.integration.sb3 import WandbCallback

import mani_skill2.envs
from chatgpt import compose_reflection_msg, parse_llm_hps, query_llm
from configs import agent_params, run_configs
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.wrappers import RecordEpisode
from tulip.envs.mani_skill2_wrapper import flatten_state_dict

# from reward_learning_with_llm_ranking import feedback, self_align
from utils import (
    create_relabel_reward_func,
    init_env,
    init_logger,
    parse_reward_func,
    parse_step_var,
    partition_and_sample,
    set_seed_everywhere,
    skill2llm_hp_ranges,
    update_llm_reward_func,
)

parser = argparse.ArgumentParser()
parser.add_argument("--skill", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--reward_option", type=str, default=None)
parser.add_argument("--iter_steps", type=int, default=100000)
parser.add_argument("--total_steps", type=int, default=16000000)
parser.add_argument("--hp_history_fn", type=str, default="hps.pkl")
parser.add_argument("--wandb_run_id", type=str, default=None)
args = parser.parse_args()


class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)

    def step(self, action):
        ob, rew, done, truncated, info = super().step(action)
        return ob, rew, False, truncated, info


class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        ob, rew, done, truncated, info = super().step(action)
        info["is_success"] = info["success"]
        return ob, rew, done, truncated, info


def make_env(cfg, max_episode_steps: int = None, record_dir: str = None):
    def _init():
        env = init_env(cfg)
        if max_episode_steps is not None:
            env = ContinuousTaskWrapper(env)
        env = SuccessInfoWrapper(env)
        if record_dir is not None:
            env = RecordEpisode(env, record_dir, info_on_video=True)
        return env

    return _init


class ReplayBufferwithRewardVars(ReplayBuffer):
    def add(self, obs, next_obs, action, reward, done, infos):
        super().add(obs, next_obs, action, reward, done, infos)

        if (self.pos == 0) and (self.full):
            self.pos = self.buffer_size - 1
        else:
            self.pos -= 1

        for env_id, info in enumerate(infos):
            if getattr(self, "reward_vars", None) is None:
                self.reward_vars = {}
            for k, v in info["reward_vars"].items():
                if k not in self.reward_vars:
                    self.reward_vars[k] = [
                        [None] * len(infos) for _ in range(self.buffer_size)
                    ]
                self.reward_vars[k][self.pos][env_id] = v

            if getattr(self, "obs_vars", None) is None:
                self.obs_vars = {}
            for k, v in info["obs_vars"].items():
                if k not in self.obs_vars:
                    self.obs_vars[k] = [
                        [None] * len(infos) for _ in range(self.buffer_size)
                    ]
                self.obs_vars[k][self.pos][env_id] = v

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True  # redudant
            self.pos = 0


class RewardLearner:
    def __init__(self, cfg):

        self.cfg = cfg

        # parse reward function
        reward_func = importlib.import_module(
            cfg.llm_reward_fn.replace("/", ".").replace(".py", "")
        )
        self.llm_reward_func = reward_func.llm_reward_func
        self.init_llm_reward_func = deepcopy(self.llm_reward_func)

        # Parse hyper-parameters
        self.llm_hps_ranges = skill2llm_hp_ranges(cfg.skill)
        self.hps, self.reward_terms, self.return_var = parse_reward_func(
            self.llm_reward_func
        )

        # load last trained record if there is
        if cfg.pretrained_fn != "":
            self.start_step = int(
                cfg.pretrained_fn.split("/")[-1]
                .split(".")[0]
                .replace("model_sac_", "")
            )
        else:
            self.start_step = int(cfg.agent_params.start_step)

        # load pre-trained llm reward functions
        self.n_iter = int(self.start_step / self.cfg.agent_params.iter_steps)
        input(f"start_step = {self.start_step}; n_iter = {self.n_iter}")
        if (self.n_iter == 0) or (cfg.reward_option == "oracle"):
            self.pref_data = []
            self.hp_history = [self.hps]
        else:
            with open(
                f"ckpts/{cfg.wandb_group}/{cfg.wandb_run_name}/pref_data.pkl",
                "rb",
            ) as fp:
                self.pref_data = pkl.load(fp)[: self.n_iter * 2]
            with open(
                f"ckpts/{cfg.wandb_group}/{cfg.wandb_run_name}/hp_history.pkl",
                "rb",
            ) as fp:
                self.hp_history = pkl.load(fp)[: self.n_iter + 1]
                self.hps = self.hp_history[-1]
                self.llm_reward_func = update_llm_reward_func(
                    self.init_llm_reward_func,
                    self.hps,
                    self.init_llm_reward_func,
                )
        self.ckpt_dir = f"ckpts/{cfg.wandb_group}/{cfg.wandb_run_name}"
        if not os.path.isdir(self.ckpt_dir):
            os.system(f"mkdir -p {self.ckpt_dir}")

    def sample_from_rollout(
        self, model, eval_envs, n_eval, n_rollout, logger, merge=False
    ):
        n_env = eval_env.num_envs
        # if (
        #    model.num_timesteps % (self.cfg.agent_params.eval_every_steps * 10)
        #    == 0
        # ):
        if False:  # True:
            for idx in range(eval_env.venv.num_envs):
                eval_env.venv.env_method(
                    method_name="set_render_image",
                    indices=idx,
                    enable_render=True,
                )
            render_on = True
        else:
            render_on = False

        ep_rewards = [[] for _ in range(n_env)]
        ep_next_obses = [[] for _ in range(n_env)]
        ep_step_vars = [[] for _ in range(n_env)]

        ep_frames = []
        ep_successes = [[] for _ in range(n_env)]
        ep_lengths = [[] for _ in range(n_env)]
        ep_oracle_rewards = [[] for _ in range(n_env)]
        ep_dists2tgt = [[] for _ in range(n_env)]
        ep_dists2goal = [[] for _ in range(n_env)]
        ep_counts = np.zeros(n_env, dtype="int")
        if merge:
            assert (
                n_eval % n_env == 0
            ), "n_eval should be integer number of n_eval_env"
            ep_count_targets = np.array([n_eval // n_env], dtype="int")
        else:
            ep_count_targets = np.array([n_eval], dtype="int")

        states = None
        episode_starts = np.ones((eval_env.num_envs,), dtype=bool)
        obs = eval_env.reset()
        while (ep_counts < ep_count_targets).any():
            actions, states = model.predict(
                obs,
                state=states,
                episode_start=episode_starts,
                deterministic=True,
            )
            next_obs, rewards, dones, infos = eval_env.step(actions)
            for env_id in range(eval_env.num_envs):
                if dones[env_id]:
                    # print(f"==> Eval episode {ep_counts.sum()} done.")
                    # ep_next_obses[env_id].append(next_obs[env_id])
                    ep_next_obses[env_id].append(infos[env_id].pop("obs_vars"))
                    ep_rewards[env_id].append(rewards[env_id])
                    ep_successes[env_id].append(infos[env_id].pop("success"))
                    ep_lengths[env_id].append(
                        infos[env_id].pop("episode_length")
                    )
                    ep_dists2tgt[env_id].append(
                        infos[env_id].pop("episode_dist2tgt")
                    )
                    ep_dists2goal[env_id].append(
                        infos[env_id].pop("episode_dist2goal")
                    )
                    ep_oracle_rewards[env_id].append(
                        infos[env_id].pop("oracle_reward")
                        / ep_lengths[env_id][-1]
                    )
                    ep_step_vars[env_id].append(
                        infos[env_id].pop("reward_vars")
                    )
                    ep_counts[env_id] += 1
                    if render_on:
                        with open(
                            glob.glob(f"{self.ckpt_dir}/*.npy")[0], "rb"
                        ) as fp:
                            ep_frames = np.load(fp)
                        for idx in range(eval_env.venv.num_envs):
                            eval_env.venv.env_method(
                                method_name="set_render_image",
                                indices=idx,
                                enable_render=False,
                            )
                        render_on = False
            obs = next_obs

        if merge:
            # unfold nested lists
            ep_next_obses = sum(ep_next_obses, [])
            ep_rewards = sum(ep_rewards, [])
            ep_successes = sum(ep_successes, [])
            ep_lengths = sum(ep_lengths, [])
            ep_dists2tgt = sum(ep_dists2tgt, [])
            ep_dists2goal = sum(ep_dists2goal, [])
            ep_oracle_rewards = sum(ep_oracle_rewards, [])
            ep_step_vars = sum(ep_step_vars, [])

            metrics = {
                "eval/avg_ep_rewards": sum(ep_rewards) / len(ep_rewards),
                "eval/avg_step_oracle_rewards": sum(ep_oracle_rewards)
                / len(ep_oracle_rewards),
                "eval/avg_ep_lengths": sum(ep_lengths) / len(ep_lengths),
                "eval/avg_ep_dists2tgt": sum(ep_dists2tgt) / len(ep_dists2tgt),
                "eval/avg_ep_dist2goal": sum(ep_dists2goal)
                / len(ep_dists2goal),
                "eval/avg_success_rate": sum(ep_successes) / len(ep_successes),
                "global_step": model.num_timesteps,
            }
            for k, v in metrics.items():
                print(f"{k}: {v}")
            video_files = glob.glob(f"{self.ckpt_dir}/*.npy")
            if len(video_files) > 0:
                metrics.update({"video": ep_frames})
                for fn in video_files:
                    os.system(f"rm {fn}")
            logger.log(metrics, model.num_timesteps)
        else:
            for env_id in range(eval_env.venv.num_envs):
                metrics = {
                    f"eval/avg_ep_rewards_{env_id}": sum(ep_rewards[env_id])
                    / len(ep_rewards[env_id]),
                    f"eval/avg_step_oracle_rewards_{env_id}": sum(
                        ep_oracle_rewards[env_id]
                    )
                    / len(ep_oracle_rewards[env_id]),
                    f"eval/avg_ep_lengths_{env_id}": sum(ep_lengths[env_id])
                    / len(ep_lengths[env_id]),
                    f"eval/avg_ep_dists2tgt_{env_id}": sum(ep_dists2tgt[env_id])
                    / len(ep_dists2tgt[env_id]),
                    f"eval/avg_ep_dist2goal_{env_id}": sum(
                        ep_dists2goal[env_id]
                    )
                    / len(ep_dists2goal[env_id]),
                    f"eval/avg_success_rate_{env_id}": sum(ep_successes[env_id])
                    / len(ep_successes[env_id]),
                    f"global_step": model.num_timesteps,
                }
                print(f"==> Eval env {env_id}:")
                for k, v in metrics.items():
                    print(f"{k}: {v}")
                video_files = glob.glob(f"{self.ckpt_dir}/*.npy")
                if len(video_files) > 0:
                    metrics.update({"video": ep_frames})
                    for fn in video_files:
                        os.system(f"rm {fn}")
                logger.log(metrics, model.num_timesteps)

            # Use rollout from env 0 for feedback
            ep_next_obses = ep_next_obses[0]
            ep_rewards = ep_rewards[0]
            ep_successes = ep_successes[0]
            ep_step_vars = ep_step_vars[0]

        ep_obs_vars = []
        ep_reward_vars = []
        for step_var in ep_step_vars:
            obs_var, reward_var = parse_step_var(
                step_var, self.hps, self.reward_terms, self.return_var
            )
            ep_obs_vars.append(obs_var)
            ep_reward_vars.append(reward_var)
        return (
            ep_rewards[:n_rollout],
            ep_next_obses[:n_rollout],
            self.hps,
            ep_obs_vars[:n_rollout],
            ep_reward_vars[:n_rollout],
            ep_successes[:n_rollout],
        )

    def sample_from_replay(
        self, model, n_recent, sort=False, start_high=True, n_bins=5
    ):
        if model.replay_buffer.pos > n_recent:
            indices = list(
                range(
                    model.replay_buffer.pos - n_recent, model.replay_buffer.pos
                )
            )
        elif model.replay_buffer.pos < n_recent and model.replay_buffer.full:
            indices = list(
                range(
                    model.replay_buffer.buffer_size - model.replay_buffer.pos,
                    model.replay_buffer.buffer_size,
                )
            ) + list(range(model.replay_buffer.pos))
        else:
            indices = list(range(model.replay_buffer.pos))

        raw_rewards = []
        raw_indices = []
        for idx in indices:
            for env_id in range(model.env.num_envs):
                if True:  # model.replay_buffer.dones[idx, env_id]:
                    raw_rewards.append(model.replay_buffer.rewards[idx, env_id])
                    raw_indices.append(
                        model.replay_buffer.buffer_size * env_id + idx
                    )
        if sort:
            raw_indices = [
                idx
                for _, idx in sorted(
                    zip(raw_rewards, raw_indices), reverse=start_high
                )
            ]
            raw_rewards = sorted(raw_rewards, reverse=start_high)
        sample_indices = partition_and_sample(raw_rewards, n_bins=n_bins)
        rewards = [raw_rewards[idx] for idx in sample_indices]
        sample_pos_indices = [raw_indices[idx] for idx in sample_indices]
        # next_obses = [
        #    model.replay_buffer.next_observations[
        #        idx % model.replay_buffer.buffer_size,
        #        idx // model.replay_buffer.buffer_size,
        #    ]
        #    for idx in sample_pos_indices
        # ]

        obs_vars = []
        reward_vars = []
        next_obses = []
        for idx in sample_pos_indices:
            next_obses.append(
                {
                    k: v[idx % model.replay_buffer.buffer_size][
                        idx // model.replay_buffer.buffer_size
                    ]
                    for k, v in model.replay_buffer.obs_vars.items()
                }
            )
            step_var = {
                k: v[idx % model.replay_buffer.buffer_size][
                    idx // model.replay_buffer.buffer_size
                ]
                for k, v in model.replay_buffer.reward_vars.items()
            }
            obs_var, reward_var = parse_step_var(
                step_var, self.hps, self.reward_terms, self.return_var
            )
            obs_vars.append(obs_var)
            reward_vars.append(reward_var)
        return rewards, next_obses, self.hps, obs_vars, reward_vars

    def relabel_reward(self, model, reward_relabel_func):
        if model.replay_buffer.full:
            idx_range = range(model.replay_buffer.buffer_size)
        else:
            idx_range = range(model.replay_buffer.pos)
        for idx in idx_range:
            for env_id in range(model.env.venv.num_envs):
                next_obs = {
                    k: v[idx][env_id]
                    for k, v in model.replay_buffer.obs_vars.items()
                }
                reward_var = {
                    k: v[idx][env_id]
                    for k, v in model.replay_buffer.reward_vars.items()
                }
                (
                    model.replay_buffer.rewards[idx][env_id],
                    _,
                ) = reward_relabel_func(next_obs, reward_var)

    def learn(
        self, model, eval_env, eval_callback, logger, hp_history=[], **kwargs
    ):
        if "oracle" in cfg.wandb_run_name:
            use_llm_reward_func = False
            llm_update = False
        else:
            use_llm_reward_func = True
            if "llm_fixed" in cfg.wandb_run_name:
                llm_update = False
                if "llm_fixed_last" in cfg.wandb_run_name:
                    self.llm_reward_func = update_llm_reward_func(
                        self.init_llm_reward_func,
                        hp_history[-1],
                        self.init_llm_reward_func,
                    )
            else:
                llm_update = True
                hp_history = hp_history

        if llm_update:
            model.replay_buffer = ReplayBufferwithRewardVars(
                model.buffer_size,
                model.observation_space,
                model.action_space,
                device=model.device,
                n_envs=model.n_envs,
                optimize_memory_usage=model.optimize_memory_usage,
                **model.replay_buffer_kwargs,
            )
        iter_steps = self.cfg.agent_params.iter_steps
        while self.start_step < self.cfg.agent_params.total_steps:

            if self.start_step % 1000000 == 0:
                model.save(f"{self.ckpt_dir}/model_sac_{self.start_step:09d}")
            print(f"==> Iter {self.n_iter}")

            if use_llm_reward_func:
                print(f"    Using reward function:\n{self.llm_reward_func}")

                # update environment reward function
                for idx in range(model.env.venv.num_envs):
                    model.env.venv.env_method(
                        method_name="setup_reward_func",
                        indices=idx,
                        reward_func=self.llm_reward_func,
                    )
                for idx in range(eval_env.venv.num_envs):
                    eval_env.venv.env_method(
                        method_name="setup_reward_func",
                        indices=idx,
                        reward_func=self.llm_reward_func,
                    )

            if llm_update:
                # reward relabeling
                relabel_reward_func = create_relabel_reward_func(
                    self.llm_reward_func
                )
                print("==> Relabelling reward...")
                self.relabel_reward(model, relabel_reward_func)
                print("==> Done")

            # train
            model.learn(
                total_timesteps=iter_steps,
                reset_num_timesteps=(self.start_step == 0),
                # callback=[eval_callback, WandbCallback(verbose=2)],
                callback=[WandbCallback(verbose=2)],
                **kwargs,
            )

            # evaluate with rollout
            (
                rollout_rewards,
                rollout_next_obses,
                rollout_init_hps,
                rollout_obs_vars,
                rollout_reward_vars,
                rollout_successes,
            ) = self.sample_from_rollout(
                model,
                eval_env,
                self.cfg.agent_params.n_eval,
                self.cfg.agent_params.n_rollout,
                logger,
                merge=(eval_env.num_envs == 1),
            )

            # 3) update llm hyper-parameters
            # 3a) If success rate exceeds 50%, keep the current reward function
            if sum(rollout_successes) / float(len(rollout_successes)) >= 0.5:
                print(
                    "success rate > 50%. Continue with the current reward func."
                )
            # 3b) Otherwise collect sample from replay and learn via self-align
            else:
                sys_msg, user_msg = compose_reflection_msg(
                    self.llm_reward_func,
                    skill,
                    rollout_obs_vars,
                    rollout_init_hps,
                )
                responses, success = query_llm(sys_msg, user_msg)
                if success:
                    print(responses)
                    llm_hps = parse_llm_hps(responses, rollout_init_hps)
                    success = llm_hps is not None
                if not success:
                    print(
                        "Automatic parsing LLM response failed. Please manually query and set llm_hps"
                    )
                    llm_hps = {}

                for k, v in llm_hps.items():
                    if k in self.hps:
                        self.hps[k] = v
                print("==> Updated hps:", self.hps)

            # update llm_reward_func with updated parameters
            with open(f"{self.ckpt_dir}/pref_data.pkl", "wb") as fp:
                pkl.dump(self.pref_data, fp)
            self.hp_history.append(self.hps)
            with open(f"{self.ckpt_dir}/hp_history.pkl", "wb") as fp:
                pkl.dump(self.hp_history, fp)
            self.llm_reward_func = update_llm_reward_func(
                self.init_llm_reward_func, self.hps, self.init_llm_reward_func
            )

            self.n_iter += 1
            self.start_step += iter_steps


if __name__ == "__main__":

    # load configs
    skill = args.skill
    assert skill in run_configs, f"Unsupported skill {skill}..."
    cfg = run_configs[skill]

    cfg.seed = args.seed
    print(f"==> Using seed {cfg.seed}")
    set_seed_everywhere(cfg.seed)

    cfg.reward_option = args.reward_option
    print(f"==> Training using {cfg.reward_option} reward")
    cfg.wandb_run_name = f"{cfg.reward_option}_stb3_{cfg.rl_algo}"
    if "text2reward" in skill:
        cfg.wandb_run_name = f"text2reward_{cfg.wandb_run_name}"
    cfg.wandb_run_name = f"{cfg.wandb_run_name}_seed{cfg.seed}"

    # n_runs = len(glob.glob(f"ckpts/{cfg.wandb_group}/{cfg.wandb_run_name}*"))
    # cfg.wandb_run_name = f"{cfg.wandb_run_name}_run_{n_runs+1}"
    cfg.agent_params = agent_params[cfg.rl_algo]

    # find existing pretrained_fn
    if args.wandb_run_id is not None:
        if not os.path.isfile(cfg.pretrained_fn):
            pretrained_files = glob.glob(
                f"ckpts/{cfg.wandb_group}/{cfg.wandb_run_name}/model_sac_*.zip"
            )
            if len(pretrained_files) > 0:
                cfg.pretrained_fn = sorted(pretrained_files)[-1]
            else:
                cfg.pretrained_fn = ""
    # initialize wandb
    logger = init_logger(cfg, args.wandb_run_id)

    # set up eval environment
    exp_name = f"{cfg.wandb_group}_{cfg.wandb_run_name}"
    eval_env = SubprocVecEnv(
        [
            make_env(cfg)  # , record_dir=f"logs/{exp_name}/videos")
            for i in range(cfg.agent_params.eval_num)
        ]
    )
    eval_env = VecMonitor(eval_env)
    eval_env.seed(cfg.seed)
    eval_env.reset()

    # set up callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"ckpts/{cfg.wandb_group}/{cfg.wandb_run_name}/best_model",
        log_path=f"ckpts/{cfg.wandb_group}/{cfg.wandb_run_name}",
        eval_freq=cfg.agent_params.eval_every_steps
        // cfg.agent_params.train_num,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )

    # set up training environment
    env = SubprocVecEnv(
        [
            make_env(cfg, max_episode_steps=cfg.task_info.max_steps)
            for i in range(cfg.agent_params.train_num)
        ]
    )
    env = VecMonitor(env)
    env.seed(cfg.seed)
    obs = env.reset()

    # set up sac algorithm
    policy_kwargs = dict(net_arch=[256, 256])
    if os.path.isfile(cfg.pretrained_fn):
        print(f"==> Loading pretrained checkpoint from {cfg.pretrained_fn}")
        model = SAC.load(cfg.pretrained_fn)
        model.set_env(env)
    else:
        print(f"==> Training from scratch.")
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            batch_size=1024,
            gamma=0.95,
            learning_starts=4000,
            ent_coef="auto_0.2",
            train_freq=8,
            gradient_steps=4,
            tensorboard_log=f"ckpts/{cfg.wandb_group}/{cfg.wandb_run_name}",
        )

    cfg.agent_params.iter_steps = args.iter_steps
    cfg.agent_params.total_steps = args.total_steps

    # for llm_fixed_last mode
    try:
        with open(args.hp_history_fn, "rb") as fp:
            hp_history = pkl.load(fp)[cfg.skill]
    except:
        hp_history = []
    reward_learner = RewardLearner(cfg)
    reward_learner.learn(model, eval_env, eval_callback, logger, hp_history)
