from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

import torch


@dataclass
class HWParams:
    wrist_cam_width: int
    wrist_cam_height: int
    wrist_cam_fx: float
    wrist_cam_fy: float
    wrist_cam_cx: float
    wrist_cam_cy: float
    wrist_cam2ref_pose: Tuple
    wrist_cam_ref_link: str
    fixed_cam_width: int
    fixed_cam_height: int
    fixed_cam_fx: float
    fixed_cam_fy: float
    fixed_cam_cx: float
    fixed_cam_cy: float
    fixed_cam2ref_pose: Tuple
    fixed_cam_ref_link: str


@dataclass
class AgentParams:
    buffer_size: int
    in_memory_buffer: bool
    device: str
    lr: float
    batch_size: int
    discount: float
    stddev_clip: float
    stddev_min: float
    stddev_max: float
    stddev_steps: int
    image_feat_dim: int
    resnet_model: str
    numerical_feat_dim: int
    hidden_dims: List[int]
    start_step: int
    iter_steps: int
    total_steps: int
    train_freq_steps: int
    gradient_steps: int
    num_expl_steps: int
    n_eval: int
    n_rollout: int
    eval_every_steps: int
    log_every_steps: int
    log_stats_window_size: int
    train_num: int
    eval_num: int


@dataclass
class ManiSkillTaskInfo:
    env_name: str
    target_object_name: str
    target_pose_key: str
    goal_pose_key: str
    distance_to_target_key: str
    distance_to_goal_key: str
    distance_to_init_key: str
    max_steps: int


agent_params = {
    "sac": AgentParams(
        buffer_size=1000000,
        in_memory_buffer=True,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
        lr=3e-4,
        batch_size=1024,
        discount=0.95,
        stddev_clip=0.3,
        stddev_min=0.1,
        stddev_max=1.0,
        stddev_steps=100000,
        image_feat_dim=512,
        resnet_model="resnet50",
        numerical_feat_dim=None,
        hidden_dims=[400, 400],
        start_step=0,
        iter_steps=100000,
        total_steps=16000000,
        train_freq_steps=8,
        gradient_steps=4,
        num_expl_steps=4000,
        n_eval=10,
        n_rollout=5,
        eval_every_steps=16000,
        log_every_steps=400,
        log_stats_window_size=100,
        train_num=8,
        eval_num=1,
    )
}

maniskill2_tasks = {
    "turn faucet": ManiSkillTaskInfo(
        env_name="TurnFaucet-v0",
        target_object_name="link_0",
        target_pose_key="wrapped_target_link_pose",
        goal_pose_key="wrapped_goal_pose",
        distance_to_target_key="wrapped_distance_to_target_object",
        distance_to_goal_key="wrapped_distance_to_goal",
        distance_to_init_key=None,
        max_steps=200,
    ),
    "pick ycb object and transport to the target position": ManiSkillTaskInfo(
        env_name="PickSingleYCB-v0",
        target_object_name=[
            "002_master_chef_can",
            "003_cracker_box",
            "004_sugar_box",
            "005_tomato_soup_can",
            "006_mustard_bottle",
            "007_tuna_fish_can",
            "008_pudding_box",
            "009_gelatin_box",
            "010_potted_meat_can",
            "011_banana",
            "012_strawberry",
            "013_apple",
            "014_lemon",
            "015_peach",
            "016_pear",
            "017_orange",
            "018_plum",
            "019_pitcher_base",
            "021_bleach_cleanser",
            "024_bowl",
            "025_mug",
            "026_sponge",
            "030_fork",
            "031_spoon",
            "032_knife",
            "033_spatula",
            "035_power_drill",
            "036_wood_block",
            "037_scissors",
            "038_padlock",
            "040_large_marker",
            "042_adjustable_wrench",
            "043_phillips_screwdriver",
            "044_flat_screwdriver",
            "048_hammer",
            "050_medium_clamp",
            "051_large_clamp",
            "052_extra_large_clamp",
            "053_mini_soccer_ball",
            "054_softball",
            "055_baseball",
            "056_tennis_ball",
            "057_racquetball",
            "058_golf_ball",
            "061_foam_brick",
            "062_dice",
            "063-a_marbles",
            "063-b_marbles",
            "065-a_cups",
            "065-b_cups",
            "065-c_cups",
            "065-d_cups",
            "065-e_cups",
            "065-f_cups",
            "065-g_cups",
            "065-h_cups",
            "065-i_cups",
            "065-j_cups",
            "070-a_colored_wood_blocks",
            "070-b_colored_wood_blocks",
            "071_nine_hole_peg_test",
            "072-a_toy_airplane",
            "072-b_toy_airplane",
            "072-c_toy_airplane",
            "072-d_toy_airplane",
            "072-e_toy_airplane",
            "073-a_lego_duplo",
            "073-b_lego_duplo",
            "073-c_lego_duplo",
            "073-d_lego_duplo",
            "073-e_lego_duplo",
            "073-f_lego_duplo",
            "073-g_lego_duplo",
            "077_rubiks_cube",
        ],
        # target_object_name="050_medium_clamp",
        target_pose_key="wrapped_obj_pose",
        goal_pose_key="wrapped_goal_pose",
        distance_to_target_key="wrapped_distance_to_target_object",
        distance_to_goal_key="wrapped_distance_to_goal",
        distance_to_init_key=None,
        max_steps=200,
    ),
    "pick cube and transport to the target position": ManiSkillTaskInfo(
        env_name="PickCube-v0",
        target_object_name="cube",
        target_pose_key="wrapped_obj_pose",
        goal_pose_key="wrapped_goal_pose",
        distance_to_target_key="wrapped_distance_to_target_object",
        distance_to_goal_key="wrapped_distance_to_goal",
        distance_to_init_key=None,
        max_steps=200,
    ),
    "insert peg into the side hole": ManiSkillTaskInfo(
        env_name="PegInsertionSide-v0",
        target_object_name="peg",
        target_pose_key="wrapped_peg_pose",
        goal_pose_key="wrapped_goal_pose",
        distance_to_target_key="wrapped_distance_to_target_object",
        distance_to_goal_key="wrapped_distance_to_goal",
        distance_to_init_key="wrapped_distance_to_init",
        max_steps=200,
    ),
    "open cabinet drawer": ManiSkillTaskInfo(
        env_name="OpenCabinetDrawer-v1",
        target_object_name="link_2",
        target_pose_key="wrapped_target_link_pose",
        goal_pose_key="wrapped_goal_pose",
        distance_to_target_key="wrapped_distance_to_target_object",
        distance_to_goal_key="wrapped_distance_to_goal",
        distance_to_init_key=None,
        max_steps=200,
    ),
    "open cabinet door": ManiSkillTaskInfo(
        env_name="OpenCabinetDoor-v1",
        target_object_name="link_0",
        target_pose_key="wrapped_target_link_pose",
        goal_pose_key="wrapped_goal_pose",
        distance_to_target_key="wrapped_distance_to_target_object",
        distance_to_goal_key="wrapped_distance_to_goal",
        distance_to_init_key=None,
        max_steps=200,
    ),
    "push a swivel chair to a target 2D location on the ground": ManiSkillTaskInfo(
        env_name="PushChair-v1",
        target_object_name="3001",
        target_pose_key="wrapped_chair_pose",
        goal_pose_key="wrapped_goal_pose",
        distance_to_target_key="wrapped_distance_to_target_object",
        distance_to_goal_key="wrapped_distance_to_goal",
        distance_to_init_key=None,
        max_steps=200,
    ),
}


@dataclass
class RunConfig:
    skill: str
    reward_func: str
    llm_reward_fn: str
    reward_option: str
    log: bool
    wandb_project: str
    wandb_group: str
    wandb_run_name: str
    hw_params: HWParams
    rl_algo: str
    use_llm_obs: bool
    replay_feedback: bool
    agent_params: AgentParams
    task_info: ManiSkillTaskInfo
    control_mode: str
    reward_mode: str
    pretrained_fn: str
    run_headless: bool
    seed: int


run_configs = {
    "text2reward pick cube and transport to the target position": RunConfig(
        skill="text2reward pick cube and transport to the target position",
        reward_func="get_maniskill2_reward",
        llm_reward_fn="reward_funcs/text2reward_pick_cube_reward_func.py",
        reward_option="oracle",
        log=False,
        wandb_project="llm_for_reward_experiment",
        wandb_group="maniskill2_pick_cube",
        wandb_run_name="llm_updated_sac",
        hw_params=None,
        rl_algo="sac",
        use_llm_obs=False,
        replay_feedback=True,
        agent_params=None,
        task_info=maniskill2_tasks[
            "pick cube and transport to the target position"
        ],
        control_mode="pd_ee_delta_pose",
        reward_mode="dense",
        pretrained_fn="",
        run_headless=True,
        seed=87,
    ),
    "pick cube and transport to the target position": RunConfig(
        skill="pick cube and transport to the target position",
        reward_func="get_maniskill2_reward",
        llm_reward_fn="reward_funcs/pick_cube_reward_func.py",
        reward_option="oracle",
        log=False,
        wandb_project="llm_for_reward_experiment",
        wandb_group="maniskill2_pick_cube",
        wandb_run_name="llm_updated_sac",
        hw_params=None,
        rl_algo="sac",
        use_llm_obs=False,
        replay_feedback=True,
        agent_params=None,
        task_info=maniskill2_tasks[
            "pick cube and transport to the target position"
        ],
        control_mode="pd_ee_delta_pose",
        reward_mode="dense",
        pretrained_fn="",
        run_headless=True,
        seed=87,
    ),
    "pick ycb object and transport to the target position": RunConfig(
        skill="pick ycb object and transport to the target position",
        reward_func="get_maniskill2_reward",
        llm_reward_fn="reward_funcs/pick_ycb_reward_func.py",
        reward_option="oracle",
        log=False,
        wandb_project="llm_for_reward_experiment",
        wandb_group="maniskill2_pick_ycb",
        wandb_run_name="llm_updated_sac",
        hw_params=None,
        rl_algo="sac",
        use_llm_obs=False,
        replay_feedback=True,
        agent_params=None,
        task_info=maniskill2_tasks[
            "pick ycb object and transport to the target position"
        ],
        control_mode="pd_ee_delta_pose",
        reward_mode="dense",
        pretrained_fn="",
        run_headless=True,
        seed=87,
    ),
    "insert peg into the side hole": RunConfig(
        skill="insert peg into the side hole",
        reward_func="get_maniskill2_reward",
        llm_reward_fn="reward_funcs/peg_insertion_reward_func.py",
        reward_option="oracle",
        log=False,
        wandb_project="llm_for_reward_experiment",
        wandb_group="maniskill2_peg_insertion",
        wandb_run_name="llm_updated_sac",
        hw_params=None,
        rl_algo="sac",
        use_llm_obs=False,
        replay_feedback=True,
        agent_params=None,
        task_info=maniskill2_tasks["insert peg into the side hole"],
        control_mode="pd_ee_delta_pose",
        reward_mode="dense",
        pretrained_fn="",
        run_headless=True,
        seed=87,
    ),
    "open cabinet drawer": RunConfig(
        skill="open cabinet drawer",
        reward_func="get_maniskill2_reward",
        llm_reward_fn="reward_funcs/open_drawer_reward_func.py",
        reward_option="oracle",
        log=False,
        wandb_project="llm_for_reward_experiment",
        wandb_group="maniskill2_open_cabinet_drawer",
        wandb_run_name="open_cabinet_drawer_oracle_sac",
        hw_params=None,
        rl_algo="sac",
        use_llm_obs=False,
        replay_feedback=True,
        agent_params=None,
        task_info=maniskill2_tasks["open cabinet drawer"],
        control_mode="base_pd_joint_vel_arm_pd_ee_delta_pose",
        reward_mode="dense",
        pretrained_fn="",
        run_headless=True,
        seed=87,
    ),
    "open cabinet door": RunConfig(
        skill="open cabinet door",
        reward_func="get_maniskill2_reward",
        llm_reward_fn="reward_funcs/open_door_reward_func.py",
        reward_option="oracle",
        log=False,
        wandb_project="llm_for_reward_experiment",
        wandb_group="maniskill2_open_cabinet_door",
        wandb_run_name="open_cabinet_door_oracle_drq_v2",
        hw_params=None,
        rl_algo="sac",
        use_llm_obs=False,
        replay_feedback=True,
        agent_params=None,
        task_info=maniskill2_tasks["open cabinet door"],
        control_mode="base_pd_joint_vel_arm_pd_ee_delta_pose",
        reward_mode="dense",
        pretrained_fn="",
        run_headless=True,
        seed=87,
    ),
    "text2reward open cabinet drawer": RunConfig(
        skill="text2reward open cabinet drawer",
        reward_func="get_maniskill2_reward",
        llm_reward_fn="reward_funcs/text2reward_open_drawer_reward_func.py",
        reward_option="oracle",
        log=False,
        wandb_project="llm_for_reward_experiment",
        wandb_group="maniskill2_open_cabinet_drawer",
        wandb_run_name="open_cabinet_drawer_oracle_sac",
        hw_params=None,
        rl_algo="sac",
        use_llm_obs=False,
        replay_feedback=True,
        agent_params=None,
        task_info=maniskill2_tasks["open cabinet drawer"],
        control_mode="base_pd_joint_vel_arm_pd_ee_delta_pose",
        reward_mode="dense",
        pretrained_fn="",
        run_headless=True,
        seed=87,
    ),
    "text2reward open cabinet door": RunConfig(
        skill="text2reward open cabinet door",
        reward_func="get_maniskill2_reward",
        llm_reward_fn="reward_funcs/text2reward_open_door_reward_func.py",
        reward_option="oracle",
        log=False,
        wandb_project="llm_for_reward_experiment",
        wandb_group="maniskill2_open_cabinet_door",
        wandb_run_name="open_cabinet_door_oracle_drq_v2",
        hw_params=None,
        rl_algo="sac",
        use_llm_obs=False,
        replay_feedback=True,
        agent_params=None,
        task_info=maniskill2_tasks["open cabinet door"],
        control_mode="base_pd_joint_vel_arm_pd_ee_delta_pose",
        reward_mode="dense",
        pretrained_fn="",
        run_headless=True,
        seed=87,
    ),
    "push a swivel chair to a target 2D location on the ground": RunConfig(
        skill="push a swivel chair to a target 2D location on the ground",
        reward_func="get_maniskill2_reward",
        llm_reward_fn="reward_funcs/push_chair_reward_func.py",
        reward_option="oracle",
        log=False,
        wandb_project="llm_for_reward_experiment",
        wandb_group="maniskill2_push_chair",
        wandb_run_name="push_chair_oracle_drq_v2",
        hw_params=None,
        rl_algo="sac",
        use_llm_obs=False,
        replay_feedback=True,
        agent_params=None,
        task_info=maniskill2_tasks[
            "push a swivel chair to a target 2D location on the ground"
        ],
        control_mode="base_pd_joint_vel_arm_pd_ee_delta_pose",
        reward_mode="dense",
        pretrained_fn="",
        run_headless=True,
        seed=87,
    ),
    "text2reward push a swivel chair to a target 2D location on the ground": RunConfig(
        skill="text2reward push a swivel chair to a target 2D location on the ground",
        reward_func="get_maniskill2_reward",
        llm_reward_fn="reward_funcs/text2reward_push_chair_reward_func.py",
        reward_option="oracle",
        log=False,
        wandb_project="llm_for_reward_experiment",
        wandb_group="maniskill2_push_chair",
        wandb_run_name="push_chair_oracle_drq_v2",
        hw_params=None,
        rl_algo="sac",
        use_llm_obs=False,
        replay_feedback=True,
        agent_params=None,
        task_info=maniskill2_tasks[
            "push a swivel chair to a target 2D location on the ground"
        ],
        control_mode="base_pd_joint_vel_arm_pd_ee_delta_pose",
        reward_mode="dense",
        pretrained_fn="",
        run_headless=True,
        seed=87,
    ),
}
