import json
import os
import time
from copy import deepcopy

import ipdb
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def compose_reflection_msg(reward_fn, skill, reward_vars, hps):

    redudant_key = [
        "success",
        "episodic_reward",
        "views",
        "action",
        "reward",
        "done",
    ]
    redudant_key += list(hps.keys())

    obs_descs = []
    for reward_var in reward_vars:
        obs_desc = ""
        for name, value in reward_var.items():
            if name in redudant_key:
                continue
            elif name.endswith("_penalty") or name.endswith("_reward"):
                continue
            else:
                if isinstance(value, float):
                    obs_desc += f" {name} = {value:.4f},"
                else:
                    obs_desc += f" {name} = {value},"
        obs_desc = obs_desc[:-1] + "."
        obs_descs.append(obs_desc)

    sys_msg = f"Given the reward function for {skill} is:\n"
    sys_msg += f"{reward_fn}\n"
    user_msg = "Given execution observations:\n"
    for i, obs_desc in enumerate(obs_descs):
        user_msg += f"  - data sample {i}:{obs_desc}\n"
    user_msg += f"Go through each data sample and check if it succeeds in executing {skill}. What action will encourage the current behavior to be more likely to successfully execute {skill}? or if this is a multi-step task, which stage the current behavior is at? what reward or penalty will prompt the current behavior to produce meaningful exploration that contains the action of the next stage?\n"
    user_msg += f"After going through all samples, count the times that each relevant reward or penalty term.  Provide chain of thought and do not use python program to analyse it.\n"
    user_msg += f"Lastly, output the identified hyper-parameter that is likely to prompt success or to the next stage behavior as a dictionary.  Please be selective to adjust the most important one and do not increase or decrease all hyper-parameters at the same time.  The key is the hyper-parameter name and the value is the recommended new value. Comment behind each to indicate if the value is suggested to increase or decrease. Do not output anything after this.\n"
    user_msg += "For example:\nResult:\n{'param_a': 1.0, 'param_b': 1.0}"

    return sys_msg, user_msg


def query_llm(
    sys_msg,
    user_msg,
    n_response=1,
    temperature=0.0,
    model_name="gpt-4",
    max_attempt=20,
):
    response = None
    for attempt in range(max_attempt):
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "text"},
                temperature=temperature,
                n=n_response,
            )
        except Exception as e:
            print(f"Attempt {attempt+1} failed with error: {e}")
            time.sleep(1)
        success = response is not None
        if success:
            break
    if not success:
        answers = []
        input("Code terminated due to too many failed attempts! Manually intervene")
        ipdb.set_trace()
    else:
        answers = [res.message.content for res in response["choices"]]
        print('[GPT4] usage_prompt_tokens:', response["usage"]["prompt_tokens"])
        print('[GPT4] usage_completion_tokens:', response["usage"]["completion_tokens"])
        print('[GPT4] usage_total_tokens:', response["usage"]["total_tokens"])
        # answers = [json.loads(res.replace("```json\n", "").replace("```", "")) for res in answers]
    return answers, success


def parse_llm_ranking(answers):
    llm_ranking = None
    for i, answer in enumerate(answers):
        lines = [l for l in answer.split("\n") if len(l) > 0]
        last_line = lines[-1]
        try:
            last_line = last_line[last_line.find('['):last_line.find(']')+1]
            llm_ranking = eval(last_line)
            break
        except:
            continue
    if llm_ranking is None:
        print("Failed to parse ranking from the last line. Manually retry")
        import ipdb; ipdb.set_trace()
    else:
        print(f"Extracting ranking from:\n{answer}")
    return llm_ranking 


def parse_llm_hps(answers, init_hps, tol=0.01):
    for i, answer in enumerate(answers):
        idx_0 = answer.find("Result:\n{")
        if idx_0 < 0:
            continue
        tmp_str = answer[idx_0 + len("Result:\n") :]
        idx_1 = tmp_str.find("}")
        if idx_1 < 0:
            continue
        try:
            llm_hp_dict = tmp_str[: idx_1 + 1]
            llm_hps = {}
            for k, v in eval(llm_hp_dict).items():
                if k not in init_hps:
                    print(f"{k} not found in hyper-parameters.")
                elif abs(init_hps[k] - v) <= tol:
                    print(f"Maintain the current value for {k}.")
                else:
                    # print(f"Setting hyper-parameter {k} to {v}.")
                    llm_hps[k] = v
            print(f"Extracting hyper-parameter adjustment from:{answer}")
            print(f"Extracted hyper-parameters:\n{llm_hps}")
            return llm_hps
        except:
            continue
    return None


if __name__ == "__main__":
    sys_msg = "You are a helpful assistant designed to output JSON."
    user_msg = "Who won the world series in 2020?"
    res, success = query_llm(sys_msg, user_msg)
    import ipdb

    ipdb.set_trace()
