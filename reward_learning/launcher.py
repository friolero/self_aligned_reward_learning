import os
import sys
from multiprocessing import Process

from configs import run_configs

idx2skill = {}
print("==> Available skills as listed:")
for i, skill in enumerate(run_configs.keys()):
    idx2skill[i] = skill
    print(f"    [{i}]: {skill}")

valid_input = False
while not valid_input:
    ipt = int(input("Please enter the skill index to be evaluated:\n"))
    valid_input = ipt in idx2skill.keys()
skill_arg = idx2skill[ipt]

idx2reward_options = {
    0: "oracle",
    1: "llm_fixed",
    2: "llm_updated",
    3: "llm_fixed_last",
}
print("==> Available run options as listed:")
for i, option in idx2reward_options.items():
    print(f"    [{i}]: {option}")

valid_input = False
while not valid_input:
    ipt = int(input("Please enter the run option index to be used:\n"))
    valid_input = ipt in idx2reward_options
reward_option_arg = idx2reward_options[ipt]

iter_steps = 100000
res = input(
    f"==> Using iter_steps = {iter_steps}. Type new value to overwrite. Enter to continue."
)
try:
    iter_steps = int(res)
except:
    iter_steps = 100000

total_steps = 16000000
res = input(
    f"==> Using total_steps = {total_steps}. Type new value to overwrite. Enter to continue."
)
try:
    total_steps = int(res)
except:
    total_steps = 16000000

procs = []
seed_args = [100, 200, 300, 400, 500]
res = input(
    f"==> Using seeds {seed_args}. Type new list to overwrite. Enter to continue."
)
try:
    seed_args = eval(res)
except:
    seed_args = [100, 200, 300, 400, 500]

for seed_arg in seed_args:
    cmd = f"python3 main.py --skill '{skill_arg}' --seed {seed_arg} --reward_option {reward_option_arg} --iter_steps {iter_steps} --total_steps {total_steps}"
    print(cmd)
    procs.append(Process(target=os.system, args=(cmd,)))
for p in procs:
    p.start()
for p in procs:
    p.join()
