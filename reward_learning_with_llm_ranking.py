import random
from copy import deepcopy
from itertools import combinations
from threading import Thread
from typing import List

import ipdb
import matplotlib.pyplot as plt
import numpy as np

import aprel

# from dotenv import load_dotenv
from chatgpt import compose_reflection_msg, parse_llm_hps, parse_llm_ranking, query_llm
from utils import denormalize_hps, normalize_hps, rank


web_api = False

class LastStep(aprel.basics.Trajectory):
    def __init__(self, obs, obs_vars):
        self.trajectory = [(obs, obs_vars)]
        self.features = [(obs, obs_vars)]
        self.clip_path = None


def describe_experience(
    reward_fn: str,
    reward_vars: List[dict],
    hps: dict = None,
    indices: List[int] = None,
    skill: str = "touch",
    describe_obs: bool = False,
    describe_reward: bool = False,
):
    redudant_key = [
        "success",
        "episodic_reward",
        "views",
        "action",
        "reward",
        "done",
    ] + list(hps.keys())

    obs_descriptions = []
    reward_descriptions = []
    for reward_var in reward_vars:
        obs_description = ""
        reward_description = ""
        for name, value in reward_var.items():
            if name in redudant_key:
                continue
            if name.endswith("_penalty") or name.endswith("_reward"):
                if isinstance(value, float):
                    reward_description += f" {name} = {value:.4f},"
                else:
                    reward_description += f" {name} = {value},"
            else:
                if isinstance(value, float):
                    obs_description += f" {name} = {value:.4f},"
                else:
                    obs_description += f" {name} = {value},"
        obs_description = obs_description[:-1] + "."
        obs_descriptions.append(obs_description)
        reward_description = reward_description[:-1] + "."
        reward_descriptions.append(reward_description)

    sys_msg = f"The reward function you provided is:\n"
    sys_msg += f"{reward_fn}\n"
    if describe_obs:
        user_msg = "Given execution observation for\n"
        for i, obs_desc in enumerate(obs_descriptions):
            user_msg += f"  - data sample {i}:{obs_desc}\n"
        user_msg += f"First summarize how many steps are there in the task.\n"
        user_msg += f"Then go through the data sample one by one and identify which stage the data sample is at.  Put object at the same step to a cluster and list all the clusters. Note one sample should only belongs to one cluster. Then rank samples from better to worse within a cluster for all cluster. If two observations are the same you can skip the comparison by excluding repeated samples index in the ranking. Lastly concatenate the ranking in a list by always putting the cluster ranking where actions are at a later step in front.\n"
        user_msg += "Make sure the last line of the reply contains and only contains the final list.\n"

        user_msg += f"Example on opening drawer task:\n"
        user_msg += " - data 1, 4 are in reaching stage, where 1 is closer than 4. The ranking for this cluster is [1, 4].\n"
        user_msg += " - data 2, 5 are in pulling stage, where 5 are pulled more than 2. The ranking for this cluster is [5, 2]\n"
        user_msg += "Pulling is at later stage than reaching. The final result is:\n"
        user_msg += "[5, 2, 1, 4]"
    else:
        sys_msg = ""
        user_msg = ""

    # not in used
    if describe_reward:
        user_msg += (
            f"\n Using the given reward function, the reward calculation for\n"
        )
        if indices is None:
            indices = list(range(len(reward_descriptions)))
        for i, obs_desc, reward_desc in zip(
            indices, obs_descriptions, reward_descriptions
        ):
            user_msg += f"  - data sample {i}: [Observations]: {obs_desc}, [Reward]: {reward_desc}\n"

    return sys_msg, user_msg


def feedback(
    skill, llm_reward_func, rewards, next_obses, init_hps, obs_vars, reward_vars
):

    reward_rank = rank(rewards)
    sys_msg, user_msg = describe_experience(
        llm_reward_func,
        obs_vars,
        hps=init_hps,
        describe_obs=True,
        skill=skill.replace("text2reward ", "").replace("eureka ", ""),
    )
    print("==> Observation description:")
    print(sys_msg)
    print(user_msg)
    if web_api:
        #llm_rank = []
        llm_rank = eval(input("Please input the output LLM ranking startswith('[') and endswith(']')."))
    else:
        responses, success = query_llm(sys_msg, user_msg)
        llm_rank = parse_llm_ranking(responses)
    print("Reward Function Rank:", reward_rank)
    print("LLM Rank:", llm_rank)

    preference_data = {
        "rewards": rewards,
        "next_obses": next_obses,
        "hps": init_hps,
        "obs_vars": obs_vars,
        "reward_vars": reward_vars,
        "reward_rank": reward_rank,
        "llm_success": [], # discarded. switched from partial order to sub-step in-cluster order
        "llm_fail_rank": llm_rank,
    }
    return preference_data


def set_tune_range(llm_hps, init_hps, llm_hp_ranges, reg_list, default_hps):
    updated_llm_hp_ranges = deepcopy(llm_hp_ranges)
    tuning_init_hps = [deepcopy(init_hps) for _ in reg_list]
    for name in updated_llm_hp_ranges:
        curr_value = init_hps[name]
        if name in llm_hps:
            llm_value = llm_hps[name]
            if llm_value == curr_value:
                updated_llm_hp_ranges[name] = [
                    curr_value - 1e-6,
                    curr_value + 1e-6,
                ]
            elif llm_value > curr_value:
                updated_llm_hp_ranges[name] = [
                    curr_value,
                    llm_hp_ranges[name][1] * abs(default_hps[name]),
                ]
                for i, reg_v in enumerate(reg_list):
                    tuning_init_hps[i][name] = abs(default_hps[name]) * min(
                        tuning_init_hps[i][name] + reg_v, llm_hp_ranges[name][1]
                    )
            else:
                updated_llm_hp_ranges[name] = [
                    llm_hp_ranges[name][0] * abs(default_hps[name]),
                    curr_value,
                ]
                for i, reg_v in enumerate(reg_list):
                    tuning_init_hps[i][name] = abs(default_hps[name]) * max(
                        tuning_init_hps[i][name] - reg_v, llm_hp_ranges[name][0]
                    )
        else:
            updated_llm_hp_ranges[name] = [curr_value - 1e-6, curr_value + 1e-6]
    return updated_llm_hp_ranges, tuning_init_hps


def aggregate_preference(preference_data):
    rewards = []
    next_obses = []
    obs_vars = []
    reward_vars = []
    all_reward_rank = []
    all_llm_success = []
    all_llm_pairs = []
    idx_offset = 0
    for iter, data in enumerate(preference_data):
        rewards += data["rewards"]
        next_obses += data["next_obses"]
        obs_vars += data["obs_vars"]
        reward_vars += data["reward_vars"]
        for idx in data["reward_rank"]:
            all_reward_rank.append(idx + idx_offset)
        all_llm_success += [idx + idx_offset for idx in data["llm_success"]]
        all_llm_pairs += gen_comparable_pairs(
            data["llm_success"] + data["llm_fail_rank"],
            data["llm_success"],
            idx_offset,
        )
        idx_offset += len(data["rewards"])
    return (
        rewards,
        next_obses,
        obs_vars,
        reward_vars,
        all_reward_rank,
        all_llm_success,
        all_llm_pairs,
    )


def gen_comparable_pairs(rank_list, incomparable_indices, idx_offset=0):
    all_pairs = []
    for pair in combinations(rank_list, 2):
        if (pair[0] in incomparable_indices) and (
            pair[1] in incomparable_indices
        ):
            continue
        all_pairs.append((pair[0] + idx_offset, pair[1] + idx_offset))
    return all_pairs


def retrieve_inconsistent_pairs(reward_rank, llm_pairs, llm_success):
    reward_idx_pairs = gen_comparable_pairs(reward_rank, llm_success)
    inconsistent_pairs = []
    for idx_pair in llm_pairs:
        if idx_pair not in reward_idx_pairs:
            inconsistent_pairs.append(idx_pair)
    return inconsistent_pairs


def learning_preference(
    get_reward,
    preference_data,
    hps,
    llm_hp_ranges,
    beta=0.9,
    n_query=5,
    patience=3,
    max_iter=5,
    log=False,
    mp_results=None,
    mp_key=None,
):
    n_iter = 0
    best_hps = hps
    (
        all_rewards,
        all_next_obses,
        all_obs_vars,
        all_reward_vars,
        all_reward_rank,
        all_llm_success,
        all_llm_pairs,
    ) = aggregate_preference(preference_data)
    min_num_inconsistent = len(all_llm_pairs)
    while True:
        inconsistent_pairs = retrieve_inconsistent_pairs(
            all_reward_rank, all_llm_pairs, all_llm_success
        )
        if len(inconsistent_pairs) <= min_num_inconsistent:
            best_hps = hps
            min_num_inconsistent = len(inconsistent_pairs)

        # Consistent case
        if len(inconsistent_pairs) == 0:
            print("==> Perfect solution found!")
            break

        if n_iter >= max_iter:
            print("==> Exceeding max searching iterations.")
            break

        # Inconsistent case, start Bayesian updating with llm ranking
        print(f"==> Iteration {n_iter} / ", mp_key)
        print("Hyperparameters:", hps)
        print("Inconsistent pairs:", inconsistent_pairs)


        # Initialization for reward learning at iter 0
        if n_iter == 0:
            last_reward_rank = None
            params = {
                "beta": beta,
                "weights": None,
                "norm_hps": np.array(
                    [v for k, v in normalize_hps(hps, llm_hp_ranges).items()]
                ),
            }
            user_model = aprel.CustomizedRewardSoftmaxUser(
                params, llm_hp_ranges, get_reward
            )
            belief = aprel.SamplingBasedBelief(user_model, [], params)
            if log:
                x = list(range(len(hps)))
                xticks = [k for k, v in hps.items()]
                plt.xticks(x, xticks)
                plt.plot(
                    x, [v for k, v in hps.items()], c=(0, 0, 1), label=f"init"
                )

        random.shuffle(all_llm_pairs)
        preference_pair = (
            inconsistent_pairs + all_llm_pairs[: n_query * len(preference_data)]
        )
        random.shuffle(preference_pair)
        for i, pair in enumerate(preference_pair):
            query = aprel.PreferenceQuery(
                [
                    LastStep(all_next_obses[pair[0]], all_obs_vars[pair[0]]),
                    LastStep(all_next_obses[pair[1]], all_obs_vars[pair[1]]),
                ]
            )
            belief.update(aprel.Preference(query, [0]))
            if False:
                print(
                    f"    ==> Iter {i}, Estimated hps: "
                    + str(belief.mean["norm_hps"])
                )

        norm_hps = {}
        for i, (k, v) in enumerate(hps.items()):
            norm_hps[k] = belief.mean["norm_hps"][i]
        hps = denormalize_hps(norm_hps, llm_hp_ranges)
        if log:
            plt.plot(
                x,
                [v for k, v in hps.items()],
                c=(0, 0, 1 - n_iter / 10),
                label=f"iter_{n_iter}",
            )

        updated_rewards = []
        for next_obs, obs_var in zip(all_next_obses, all_obs_vars):
            inputs = hps.copy()
            inputs["obs"] = next_obs
            inputs["reward_vars"] = obs_var
            updated_rewards.append(get_reward(**inputs)[0])

        updated_reward_rank = []
        idx_offset = 0
        for data in preference_data:
            iter_len = len(data["reward_rank"])
            iter_reward_rank = rank(
                updated_rewards[idx_offset : idx_offset + iter_len]
            )
            iter_reward_rank = [idx + idx_offset for idx in iter_reward_rank]
            updated_reward_rank += iter_reward_rank
            idx_offset += iter_len

        if last_reward_rank is None:
            n_repeat = 0
        else:
            if updated_reward_rank == last_reward_rank:
                n_repeat += 1
                if n_repeat >= patience:
                    print("Best hyperparameter found:", best_hps)
                    break
            else:
                n_repeat = 0
        # all_rewards = updated_rewards
        all_reward_rank = updated_reward_rank
        last_reward_rank = updated_reward_rank
        n_iter += 1

    if log:
        plt.show()

    if mp_results is not None:
        mp_results[mp_key] = {
            "best_hps": best_hps,
            "n_inconsistent": min_num_inconsistent,
        }
    else:
        return {"best_hps": best_hps, "n_inconsistent": min_num_inconsistent}


def select_furthest_best_hps(search_results):
    opt_reg = None
    min_inconsistent = 1e6
    reg_list = sorted(list(search_results.keys()))
    for reg in reg_list:
        if search_results[reg]["n_inconsistent"] <= min_inconsistent:
            opt_reg = reg
            best_hp = search_results[reg]["best_hps"]
            min_inconsistent = search_results[reg]["n_inconsistent"]
    print(f"==> Best hps chosen at {opt_reg}: ")
    print(best_hp)
    return best_hp, min_inconsistent


def select_nearest_best_hps(search_results):
    opt_reg = None
    min_inconsistent = 1e6
    reg_list = sorted(list(search_results.keys()))
    for reg in reg_list:
        if search_results[reg]["n_inconsistent"] < min_inconsistent:
            opt_reg = reg
            best_hp = search_results[reg]["best_hps"]
            min_inconsistent = search_results[reg]["n_inconsistent"]
    print(f"==> Best hps chosen at {opt_reg}: ")
    print(best_hp)
    return best_hp, min_inconsistent


def learning_preference_with_reg(
    get_reward,
    preference_data,
    reg_list,
    init_hps,
    llm_hp_ranges,
    tuning,
    tuning_init_hps,
    hp_furthest=True,
):
    """multi-processing with different distance regularization."""
    search_results = {}
    procs = []
    default_hps = preference_data[0]["hps"]
    for i, reg in enumerate(reg_list):
        print(f"==> Reg {reg}")
        tmp_llm_hp_ranges = {
            k: [
                max(llm_hp_ranges[k][0], v - reg * default_hps[k]),
                min(llm_hp_ranges[k][1], v + reg * default_hps[k]),
            ]
            for k, v in init_hps.items()
        }
        if tuning:
            assert tuning_init_hps is not None, "Please provide tuning_init_hps"
            hps = tuning_init_hps[i]
        else:
            hps = {
                k: np.random.uniform(
                    tmp_llm_hp_ranges[k][0], tmp_llm_hp_ranges[k][1]
                )
                for k, v in init_hps.items()
            }
        print("Hp ranges:", tmp_llm_hp_ranges)
        if tuning:
            n_aggregate = 2  # len(preference_data)
        else:
            n_aggregate = 2  # len(preference_data)
        mp = len(reg_list) > 1
        if mp:
            procs.append(
                Thread(
                    target=learning_preference,
                    args=(
                        get_reward,
                        preference_data[-n_aggregate:],
                        hps,
                        tmp_llm_hp_ranges,
                        0.9,
                        5,
                        3,
                        5,
                        False,
                        search_results,
                        reg,
                    ),
                )
            )
        else:
            search_results[reg] = learning_preference(
                get_reward,
                preference_data[-n_aggregate:],
                hps,
                tmp_llm_hp_ranges,
            )
            if (search_results[reg]["n_inconsistent"] == 0) and (not tuning):
                break
    if mp:
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        print("Finished with preference update:", search_results)
    if tuning:
        if hp_furthest:
            best_hps, min_inconsistent = select_furthest_best_hps(search_results)
        else:
            best_hps, min_inconsistent = select_nearest_best_hps(search_results)
    else:
        best_hps, min_inconsistent = select_nearest_best_hps(search_results)
    return best_hps, min_inconsistent


def self_align(
    skill,
    llm_reward_func,
    preference_data,
    llm_hp_ranges,
    get_reward,
    reg_list=[1, 3, 5, 10],
    consistent_unchanged=False,
):

    init_hps = preference_data[-1]["hps"]
    default_hps = preference_data[0]["hps"]
    llm_success = preference_data[-1]["llm_success"]
    llm_fail_rank = preference_data[-1]["llm_fail_rank"]
    reward_rank = preference_data[-1]["reward_rank"]

    llm_pairs = gen_comparable_pairs(llm_success + llm_fail_rank, llm_success)
    inconsistent_pairs = retrieve_inconsistent_pairs(
        reward_rank, llm_pairs, llm_success
    )

    # If consistent. Tuning in parameter and direction guided with llm.
    if len(inconsistent_pairs) == 0:

        if consistent_unchanged:
            print("No discrepency is found. Keep hps the same.")
            llm_hps = {}
        else:
            print("No discrepency is found. Query hp adjustment to GPT")
            sys_msg, user_msg = compose_msg(
                llm_reward_func,
                skill.replace("text2reward", "").replace("eureka", ""),
                preference_data[-1]["obs_vars"],
                init_hps,
            )
            print(sys_msg)
            print(user_msg)
            if web_api:
                llm_hps = eval(input("Please enter the output llm hp adjustments startswith('{') and endswith('}')."))
            else:
                responses, success = query_llm(sys_msg, user_msg)
                if success:
                    llm_hps = parse_llm_hps(responses, init_hps)
                    success = llm_hps is not None
                if not success:
                    print(
                        "Automatic parsing LLM response failed. Please manually query and set llm_hps"
                    )
                    llm_hps = {}
                    ipdb.set_trace()

        llm_hp_ranges, tuning_init_hps = set_tune_range(
            llm_hps, init_hps, llm_hp_ranges, reg_list, default_hps,
        )

        tuning = True
    else:
        tuning = False
        tuning_init_hps = None
    updated_hps, min_inconsistent = learning_preference_with_reg(
        get_reward,
        preference_data,
        reg_list,
        init_hps,
        llm_hp_ranges,
        tuning,
        tuning_init_hps,
    )
    if min_incosistent < len(inconsistent_pairs):
        return updated_hps
    else:
        print("No better solution found. Still use the current hps.")
        return preference_data[-1]["hps"]
    return init_hps
