import time
import types
import uuid
from copy import deepcopy
from typing import Callable, List, Tuple, Union

import gymnasium as gym
import numpy as np
import wandb
from gymnasium import spaces
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.common import get_dtype_bounds
from mani_skill2.utils.geometry import transform_points
from mani_skill2.utils.sapien_utils import (
    get_entity_by_name,
    get_pairwise_contact_impulse,
)
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

from transform_utils import (
    euler2quat,
    pos_quat2pose_matrix,
    pose_matrix2pos_quat,
    quat2euler,
)


def flatten_state_dict(state_dict: dict) -> np.ndarray:
    """Flatten a dictionary containing states recursively.

    Args:
        state_dict: a dictionary containing scalars or 1-dim vectors.

    Raises:
        AssertionError: If a value of @state_dict is an ndarray with ndim > 2.

    Returns:
        np.ndarray: flattened states.

    Notes:
        The input is recommended to be ordered (e.g. OrderedDict).
        However, since python 3.7, dictionary order is guaranteed to be insertion order.
    """
    states = []
    for key, value in state_dict.items():
        if key in ["camera_param", "image"] or key.startswith("wrapped_"):
            continue
        if isinstance(value, dict):
            state = flatten_state_dict(value)
            if state.size == 0:
                state = None
        elif isinstance(value, (tuple, list)):
            state = None if len(value) == 0 else value
        elif isinstance(value, (bool, np.bool_, int, np.int32, np.int64)):
            # x = np.array(1) > 0 is np.bool_ instead of ndarray
            state = int(value)
        elif isinstance(value, (float, np.float32, np.float64)):
            state = np.float32(value)
        elif isinstance(value, np.ndarray):
            if value.ndim > 2:
                raise AssertionError(
                    "The dimension of {} should not be more than 2.".format(key)
                )
            state = value if value.size > 0 else None
        else:
            raise TypeError("Unsupported type: {}".format(type(value)))
        if state is not None:
            states.append(state)
    if len(states) == 0:
        return np.empty(0)
    else:
        return np.hstack(states)


class ManiSkill2RobotWrapper:
    def __init__(
        self,
        env,
        finger_links=["panda_leftfinger", "panda_rightfinger"],
        gripper_links=["panda_hand", "panda_leftfinger", "panda_rightfinger"],
    ):
        self._env = env
        self._link_actors = self._env.agent.robot.get_links()
        self._link_names = [link.name for link in self._link_actors]
        self._gripper_actors = [
            get_entity_by_name(self._env.agent.robot.get_links(), name)
            for name in gripper_links
        ]
        self._finger_actors = [
            get_entity_by_name(self._env.agent.robot.get_links(), name)
            for name in finger_links
        ]

        init_state = self._env.agent.get_state()
        self._base_pos = init_state["robot_root_pose"].p
        self._base_orn = init_state["robot_root_pose"].q[[1, 2, 3, 0]]
        self._base2world = pos_quat2pose_matrix(self._base_pos, self._base_orn)

    def _parse_robot(self):
        self._num_joints = len(self._env.agent.robot_link_ids)

    @property
    def links(self):
        return self._link_actors

    @property
    def link_names(self):
        return self._link_names

    @property
    def num_joint(self):
        return self._num_joints

    @property
    def ee_pose(self):
        ee_pose = self._env.tcp.pose
        return (ee_pose.p, R.from_quat(ee_pose.q[[1, 2, 3, 0]]).as_euler("xyz"))

    @property
    def q(self):
        return self._env.agent.get_proprioception()["qpos"]

    @property
    def dq(self):
        return self._env.agent.get_proprioception()["qvel"]

    @property
    def joint_reaction_forces(self):
        return np.zeros_like(self.q)

    def ignore_collision(self, contact):
        if (
            contact.actor0.name.endswith("panda_hand")
            and contact.actor1.name.startswith("panda")
        ) or (
            contact.actor0.name.startswith("panda")
            and contact.actor1.name.endswith("panda_hand")
        ):
            return True
        else:
            return False

    @property
    def ee_reaction_force(self):
        ee_contact_forces = [np.zeros(3) for _ in self._gripper_actors]
        for contact in self._env.agent.scene.get_contacts():
            if self.ignore_collision(contact):
                continue
            impulse = np.sum([p.impulse for p in contact.points], axis=0)
            for i, gripper_actor in enumerate(self._gripper_actors):
                if contact.actor0.name == gripper_actor.name:
                    ee_contact_forces[i] += impulse
                elif contact.actor1.name == gripper_actor.name:
                    ee_contact_forces[i] -= impulse
        net_contact_force = sum(ee_contact_forces)
        return net_contact_force

    def world2base(self, pos, quat):
        to_base = self._env.agent.robot.pose.inv()
        to_base = pos_quat2pose_matrix(to_base.p, to_base.q[[1, 2, 3, 0]])
        pose_w = pos_quat2pose_matrix(pos, quat)
        pose_b = np.matmul(to_base, pose_w)
        pos, quat = pose_matrix2pos_quat(pose_b)
        return pos, quat

    def gripper_in_contact(self, obj, min_impulse=1e-6):
        contacts = self._env.agent.scene.get_contacts()
        gimpulses = []
        for actor in self._gripper_actors:
            if isinstance(obj, list):
                gimpulses.append(
                    sum(
                        [
                            get_pairwise_contact_impulse(contacts, actor, o)
                            for o in obj
                        ]
                    )
                )
            else:
                gimpulses.append(
                    get_pairwise_contact_impulse(contacts, actor, obj)
                )
        contacted = (np.array(gimpulses) >= min_impulse).any()
        return contacted

    def check_ee_alignment(self, obj):
        assert len(obj) == 1, "No more than one object to check alignment"
        ee_pose_wrt_obj = obj[0].pose.inv() * self._env.tcp.pose
        ee_rot_wrt_obj = ee_pose_wrt_obj.to_transformation_matrix()[:3, :3]
        gt_rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        # gt_rot_2 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        orn_err = (
            R.from_matrix(gt_rot).as_euler("xyz")
            - R.from_matrix(ee_rot_wrt_obj).as_euler("xyz").tolist()
        )
        orn_err = np.array([v if v < np.pi else v - 2 * np.pi for v in orn_err])
        return orn_err

    def is_grasping(self, obj, max_angle=20):
        if isinstance(obj, list):
            grasped = np.any(
                [
                    self._env.agent.check_grasp(o, max_angle=max_angle)
                    for o in obj
                ]
            )
        else:
            grasped = self._env.agent.check_grasp(obj, max_angle=max_angle)
        return grasped


class ManiSkill2DualArmRobotWrapper:
    def __init__(
        self,
        env,
        finger_links=["panda_leftfinger", "panda_rightfinger"],
        gripper_links=["panda_hand", "panda_leftfinger", "panda_rightfinger"],
    ):
        self._env = env
        self._link_actors = self._env.agent.robot.get_links()
        self._link_names = [link.name for link in self._link_actors]

        init_state = self._env.agent.get_state()
        self._base_pos = init_state["robot_root_pose"].p
        self._base_orn = init_state["robot_root_pose"].q[[1, 2, 3, 0]]
        self._base2world = pos_quat2pose_matrix(self._base_pos, self._base_orn)

        self._left_arm = ManiSkill2RobotWrapper(
            env=env,
            finger_links=["left_" + ln for ln in finger_links],
            gripper_links=["left_" + ln for ln in gripper_links],
        )
        self._right_arm = ManiSkill2RobotWrapper(
            env=env,
            finger_links=["right_" + ln for ln in finger_links],
            gripper_links=["right_" + ln for ln in gripper_links],
        )

    def _parse_robot(self):
        self._num_joints = len(self._env.agent.robot_link_ids)

    @property
    def left_arm(self):
        return self._left_arm

    @property
    def right_arm(self):
        return self._right_arm

    @property
    def links(self):
        return self._link_actors

    @property
    def link_names(self):
        return self._link_names

    @property
    def num_joint(self):
        return self._num_joints

    @property
    def left_ee_pose(self):
        ee_pose = self._env.left_tcp.pose
        return (ee_pose.p, R.from_quat(ee_pose.q[[1, 2, 3, 0]]).as_euler("xyz"))

    @property
    def right_ee_pose(self):
        ee_pose = self._env.right_tcp.pose
        return (ee_pose.p, R.from_quat(ee_pose.q[[1, 2, 3, 0]]).as_euler("xyz"))

    @property
    def ee_pose(self):
        left_ee_pos, left_ee_orn = self.left_ee_pose
        right_ee_pos, right_ee_orn = self.right_ee_pose
        return (left_ee_pos, left_ee_orn, right_ee_pos, right_ee_orn)

    @property
    def q(self):
        return self._env.agent.get_proprioception()["qpos"]

    @property
    def dq(self):
        return self._env.agent.get_proprioception()["qvel"]

    @property
    def joint_reaction_forces(self):
        return np.zeros_like(self.q)

    @property
    def left_ee_reaction_force(self):
        return self._left_arm.ee_reaction_force

    @property
    def right_ee_reaction_force(self):
        return self._right_arm.ee_reaction_force

    @property
    def ee_reaction_force(self):
        left_net_contact_force = self._left_arm.ee_reaction_force
        right_net_contact_force = self._right_arm.ee_reaction_force
        return np.concatenate((left_net_contact_force, right_net_contact_force))

    def world2base(self, pos, quat):
        to_base = self._env.agent.robot.pose.inv()
        to_base = pos_quat2pose_matrix(to_base.p, to_base.q[[1, 2, 3, 0]])
        pose_w = pos_quat2pose_matrix(pos, quat)
        pose_b = np.matmul(to_base, pose_w)
        pos, quat = pose_matrix2pos_quat(pose_b)
        return pos, quat

    def gripper_in_contact(self, obj, min_impulse=1e-6, any=True):
        left_contacted = self._left_arm.gripper_in_contact(obj, min_impulse)
        right_contacted = self._right_arm.gripper_in_contact(obj, min_impulse)
        if any:
            return left_contacted or right_contacted
        else:
            return left_contacted, right_contacted

    def is_grasping(self, obj, max_angle=20, any=True):
        left_grasping = self._left_arm.is_grasping(obj, max_angle)
        right_grasping = self._right_arm.is_grasping(obj, max_angle)
        if any:
            return left_grasping or right_grasping
        else:
            return left_grasping, right_grasping


class ManiSkill2FixedBaseTaskWrapper(gym.Wrapper):
    def __init__(
        self,
        env_name: str,
        target_object_name: str,
        target_pose_key: str,
        goal_pose_key: str,
        distance_to_target_key: str,
        distance_to_goal_key: str,
        distance_to_init_key: str,
        obs_mode: str = "state_dict",
        reward_mode: str = "sparse",
        terminal_at_success: bool = True,
        control_mode: str = "pd_ee_pose",
        action_repeat: int = 3,
        max_steps: int = 25,
        gripper_init_action: bool = 1,
        relevant_obs_names: List[str] = None,
        cam_name_map: dict = {"base_camera": "front", "hand_camera": "wrist"},
        reward_func: Callable = None,
        render_image: bool = False,
        ckpt_dir: str = "",
    ) -> None:

        self._render_image = render_image
        self._obs_mode = obs_mode
        self._env = gym.make(
            env_name,
            obs_mode=obs_mode,
            reward_mode=reward_mode,
            # reward_mode="dense",
            control_mode=control_mode,
            render_mode="cameras",
            max_episode_steps=max_steps,
        )
        super().__init__(self._env)
        self._target_object_name = target_object_name

        self._cam_name_map = cam_name_map
        self._relevant_obs_names = relevant_obs_names
        self._target_pose_key = target_pose_key
        self._goal_pose_key = goal_pose_key
        self._distance_to_target_key = distance_to_target_key
        self._distance_to_goal_key = distance_to_goal_key
        self._distance_to_init_key = distance_to_init_key
        assert (
            "joint_reaction_forces" not in self._relevant_obs_names
        ), "Unsupported observation joint_reaction_forces."
        assert (
            "object_poses" not in self._relevant_obs_names
        ), "Unsupported observation object_poses."

        obs, info = self._env.reset(seed=2022, options=dict(reconfigure=True))
        state_obs = flatten_state_dict(obs)
        if False:
            self.observation_space = {
                "original_obs": spaces.Box(
                    low=-np.inf, high=np.inf, shape=state_obs.shape
                ),
                "action": self._env.action.space,
            }
            self.update_obs_space_from_scene()
            self.update_obs_space_from_robot()
            if obs_mode == "image":
                self.update_obs_space_from_camera()
            if self._relevant_obs_names is None:
                self._relevant_obs_names = list(self.observation_space.keys())
            self.observation_space = spaces.Dict(
                {
                    k: v
                    for k, v in self.observation_space.items()
                    if k in self._relevant_obs_names
                }
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=state_obs.shape
            )

        self.action_space = self._env.action_space

        self._terminal_at_success = terminal_at_success
        if reward_func is not None:
            self.setup_reward_func(reward_func)
        else:
            self.get_reward = self.get_maniskill2_reward

        self._ckpt_dir = ckpt_dir
        self._n_episode = -1
        self.max_steps = max_steps  # self._env.max_episode_steps
        _ = self.reset()

    @property
    def sapien(self):
        return self._env

    def get_world_pcd(self, obj, pcd):
        return transform_points(obj.pose.to_transformation_matrix(), pcd)

    def reset(self, seed=None, options=None):
        self._env.reset(seed=seed, options=options)
        self._scene = self._env.agent.scene
        self.robot = ManiSkill2RobotWrapper(self._env)

        all_actors = self._scene.get_all_actors()
        target_object_name = self._target_object_name
        for articulations in self._scene.get_all_articulations():
            if articulations.name == self._target_object_name:
                target_object_name = [l.name for l in articulations.get_links()]
            all_actors += articulations.get_links()
        self.objects = []
        self.target_object = []
        for actor in all_actors:
            if actor.name in self.robot.link_names:
                continue
            elif actor.name in target_object_name:
                self.target_object.append(actor)
            else:
                self.objects.append(actor)

        self._step_num = 0
        self._n_episode += 1
        self._episodic_reward = 0
        self._episodic_oracle_reward = 0
        self._episode_dist2tgt = 0
        self._episode_dist2goal = 0
        self._maniskill2_reward = 0
        self._maniskill2_info = {}
        self._views = {
            self._cam_name_map[name]: [] for name in self._cam_name_map
        }

        obs = self.get_observation()
        return obs["original_obs"], {}

    def update_obs_space_from_scene(self):
        scene_observation_space = {
            "target_object_pose": spaces.Box(low=-10, high=10, shape=(6,)),
            "goal_pose": spaces.Box(low=-10, high=10, shape=(6,)),
            "object_poses": spaces.Box(low=-10, high=10, shape=(6,)),
            "distance_to_target": spaces.Box(low=-10, high=10, shape=(3,)),
            "distance_to_init": spaces.Box(low=-10, high=10, shape=(6,)),
            "distance_to_goal": spaces.Box(low=-10, high=10, shape=(3,)),
            "contacted": spaces.Box(low=-0, high=1, shape=(2,)),
            "collided": spaces.Box(low=-0, high=1, shape=(2,)),
        }
        self.observation_space.update(scene_observation_space)

    def update_obs_space_from_robot(self):
        robot_observation_space = {
            "joint_positions": self._env.observation_space["agent"]["qpos"],
            "joint_velocities": self._env.observation_space["agent"]["qvel"],
            "joint_reaction_forces": spaces.Box(
                low=-1000,
                high=100,
                shape=self._env.observation_space["agent"]["qpos"].shape,
            ),
            "ee_pose": spaces.Box(low=-10, high=10, shape=(6,)),
            "ee_reaction_force": spaces.Box(low=-1000, high=1000, shape=(3,)),
        }
        self.observation_space.update(robot_observation_space)

    def update_obs_space_from_camera(self):
        for name in self._env.observation_space["image"]:
            cam_name = self._cam_name_map[name]
            self.observation_space.update(
                {
                    f"{cam_name}_rgb": self._env.observation_space["image"][
                        name
                    ]["Color"],
                    f"{cam_name}_depth": spaces.Box(
                        low=0,
                        high=65.535,
                        shape=self._env.observation_space["image"][name][
                            "Position"
                        ].shape[:-1]
                        + (3,),
                    ),
                }
            )

    @property
    def render_image(self):
        return self._render_image

    def set_render_image(self, enable_render: bool):
        self._render_image = enable_render

    def get_observation(self, obs=None):
        if obs is None:
            obs = self._env.get_obs()
        state_obs = flatten_state_dict(obs)
        # robot observation
        observation = {
            "original_obs": state_obs,
            "joint_positions": obs["agent"]["qpos"],
            "joint_velocities": obs["agent"]["qvel"],
            "joint_reaction_forces": np.empty(obs["agent"]["qpos"].shape),
            "ee_pose": np.concatenate(self.robot.ee_pose),
            "ee_reaction_force": self.robot.ee_reaction_force,
            "action": self.action_space.sample().fill(0),
        }
        # camera observation
        if self._obs_mode == "image":
            for name in obs["image"]:
                rgb = obs["image"][name]["Color"][..., :3] * 255
                rgb = rgb.astype(np.uint8)
                depth = obs["image"][name]["Position"][..., 2]
                depth = (depth * 1000).astype(np.uint16)
                depth = depth[..., np.newaxis].repeat(3, axis=-1)
                cam_name = self._cam_name_map[name]
                observation.update(
                    {f"{cam_name}_rgb": rgb, f"{cam_name}_depth": depth}
                )
                self._views[cam_name].append(rgb)
        elif self._render_image:
            while True:
                try:
                    rgb = self._env.render()
                    break
                except:
                    print("Rendering error, Rerender in 1s...")
                    time.sleep(1.0)
            for cam_name in self._views:
                self._views[cam_name].append(rgb)

        # scene observation
        observation["object_poses"] = np.empty(6)
        if self._target_pose_key is not None:
            observation["target_object_pose"] = obs["extra"][
                self._target_pose_key
            ]
        else:
            observation["target_object_pose"] = np.empty(6)
        if self._distance_to_target_key is None:
            observation["distance_to_target"] = (
                observation["ee_pose"][:3] - observation["target_object_pos"]
            )
        else:
            observation["distance_to_target"] = obs["extra"][
                self._distance_to_target_key
            ]

        if self._distance_to_init_key is not None:
            observation["distance_to_init"] = obs["extra"][
                self._distance_to_init_key
            ]
        else:
            observation["distance_to_init"] = np.empty(6)

        if self._goal_pose_key is not None:
            observation["goal_pose"] = obs["extra"][self._goal_pose_key]
        else:
            observation["goal_pose"] = np.empty(6)

        if self._distance_to_goal_key is not None:
            observation["distance_to_goal"] = obs["extra"][
                self._distance_to_goal_key
            ]
        else:
            observation["distance_to_goal"] = np.empty((3,))

        if self.robot.gripper_in_contact(self.target_object):
            observation["contacted"] = np.array([1.0, 0.0])
        else:
            observation["contacted"] = np.array([0.0, 1.0])

        if self.detect_collision():
            observation["collided"] = np.array([1.0, 0.0])
        else:
            observation["collided"] = np.array([0.0, 1.0])

        # filter relevant observations only
        relevant_obs = {
            k: v
            for k, v in observation.items()
            if k in self._relevant_obs_names
        }
        return relevant_obs

    @property
    def camera_views(self):
        return {name: self._views[name][-1] for name in self._views}

    def step(self, action: np.ndarray, log=False):

        (
            next_obs,
            self._maniskill2_reward,
            done,
            _,
            self._maniskill2_info,
        ) = self._env.step(action)
        if log:
            print("    Reached pose:", self.robot.ee_pose[0])

        self._step_num += 1
        next_obs = self.get_observation()
        next_obs["action"] = action
        reward, reward_vars = self.get_reward(next_obs)
        done = self.get_termination(done)

        self._episodic_reward += reward
        self._episodic_oracle_reward += self._maniskill2_reward
        info = {"success": self._maniskill2_info["success"]}
        truncated = self._step_num >= self.max_steps
        obs_vars = deepcopy(next_obs)
        obs_vars.pop("original_obs")
        info["obs_vars"] = obs_vars
        info["reward_vars"] = reward_vars
        if done:
            self._episode_dist2tgt += np.linalg.norm(
                next_obs["distance_to_target"]
            )
            self._episode_dist2goal += np.linalg.norm(
                next_obs["distance_to_goal"]
            )
            if len(self._views["front"]) > 0:
                with open(
                    f"{self._ckpt_dir}/{uuid.uuid3(uuid.NAMESPACE_DNS, str(self._episodic_reward))}.npy",
                    "wb",
                ) as fp:
                    np.save(
                        fp,
                        np.stack(self._views["front"]).transpose([0, 3, 1, 2]),
                    )
            info.update(
                {
                    "episodic_reward": self._episodic_reward,
                    "episode_length": self._step_num,
                    "episode_dist2tgt": self._episode_dist2tgt,
                    "episode_dist2goal": self._episode_dist2goal,
                    "oracle_reward": self._episodic_oracle_reward,
                }
            )

        return next_obs["original_obs"], reward, done, truncated, info

    def set_terminal_at_success(self, terminal_at_success: bool):
        self._terminal_at_success = terminal_at_success

    def get_termination(self, done) -> bool:
        if self._terminal_at_success and (
            done or (self._step_num >= self.max_steps)
        ):
            return True
        elif (not self._terminal_at_success) and (
            self._step_num >= self.max_steps
        ):
            return True
        else:
            return False

    def get_maniskill2_reward(self, obs) -> float:
        return self._maniskill2_reward, self._maniskill2_info

    def setup_reward_func(self, reward_func: str) -> None:
        # if function being passed
        if "def " in reward_func:
            llm_reward_func = reward_func.replace("(obs,", "(self, obs,")
            llm_reward_func = llm_reward_func.replace(
                "ee_reaction_forece", "ee_reaction_force"
            )
            llm_reward_func = llm_reward_func.replace("env.", "self.")
            llm_reward_func = llm_reward_func.replace(
                "obj.in_contact(self.robot)",
                "self.robot.gripper_in_contact(obj)",
            )
            llm_reward_func = llm_reward_func.replace(
                "return total_reward",
                "vars = {}\n    for k, v in locals().items():\n        if type(v) in [np.int, np.float32, np.float64, int, float, bool, np.bool_]:\n            vars[k]=v\n    return total_reward, vars\n",
            )
            exec(llm_reward_func)
            self.get_reward = types.MethodType(eval("get_reward"), self)
        # if existing function name being passed
        else:
            self.get_reward = getattr(self, reward_func)

    def detect_collision(self, ignore_target=True):
        force_magnitude = np.linalg.norm(self.robot.ee_reaction_force[:3])
        if force_magnitude <= 0:
            return False
        elif self.robot.gripper_in_contact(self.objects):
            return True
        elif self.robot.gripper_in_contact(self.target_object):
            if ignore_target:
                return False
            else:
                return True
        else:
            return False  # True

    def detect_success(self, obs):
        return self._maniskill2_info["success"]


class ManiSkill2MobileBaseTaskWrapper(gym.Wrapper):
    def __init__(
        self,
        env_name: str,
        target_object_name: str,
        target_pose_key: str,
        goal_pose_key: str,
        distance_to_target_key: str,
        distance_to_goal_key: str,
        distance_to_init_key: str,
        obs_mode: str = "state_dict",
        reward_mode: str = "dense",
        terminal_at_success: bool = True,
        control_mode: str = "base_pd_joint_vel_arm_pd_ee_delta_pose",
        max_steps: int = 25,
        relevant_obs_names: List[str] = None,
        reward_func: Callable = None,
        render_image: bool = False,
        ckpt_dir: str = "",
    ) -> None:
        self._render_image = render_image
        self._obs_mode = obs_mode
        self._env = gym.make(
            env_name,
            obs_mode=self._obs_mode,
            reward_mode=reward_mode,
            control_mode=control_mode,
            render_mode="cameras",
            max_episode_steps=max_steps,
        )
        super().__init__(self._env)
        self._target_object_name = target_object_name

        self._relevant_obs_names = relevant_obs_names
        self._target_pose_key = target_pose_key
        self._goal_pose_key = goal_pose_key
        self._distance_to_target_key = distance_to_target_key
        self._distance_to_goal_key = distance_to_goal_key
        self._distance_to_init_key = distance_to_init_key
        assert (
            "joint_reaction_forces" not in self._relevant_obs_names
        ), "Unsupported observation joint_reaction_forces."
        assert (
            "object_poses" not in self._relevant_obs_names
        ), "Unsupported observation object_poses."

        obs, info = self._env.reset(seed=2022)
        state_obs = flatten_state_dict(obs)
        if False:
            self.observation_space = {
                "original_obs": spaces.Box(
                    low=-np.inf, high=np.inf, shape=state_obs.shape
                ),
                "action": self._env.action.space,
            }
            self.update_obs_space_from_scene()
            self.update_obs_space_from_robot()
            if self._obs_mode == "image":
                self.update_obs_space_from_camera()
            if self._relevant_obs_names is None:
                self._relevant_obs_names = list(self.observation_space.keys())
            self.observation_space = spaces.Dict(
                {
                    k: v
                    for k, v in self.observation_space.items()
                    if k in self._relevant_obs_names
                }
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=state_obs.shape
            )

        self.action_space = self._env.action_space

        self._terminal_at_success = terminal_at_success
        if reward_func is not None:
            self.setup_reward_func(reward_func)
        else:
            self.get_reward = self.get_maniskill2_reward

        self._ckpt_dir = ckpt_dir
        self._n_episode = -1
        self.max_steps = max_steps  # self._env.max_episode_steps
        _ = self.reset()

    @property
    def sapien(self):
        return self._env

    def get_world_pcd(self, obj, pcd):
        return transform_points(obj.pose.to_transformation_matrix(), pcd)

    def reset(self, seed=None, options=None):
        self._env.reset(seed=seed, options=options)
        self._scene = self._env.agent.scene
        self.robot = ManiSkill2RobotWrapper(
            self._env,
            finger_links=["right_panda_leftfinger", "right_panda_rightfinger"],
            gripper_links=[
                "right_panda_hand",
                "right_panda_leftfinger",
                "right_panda_rightfinger",
            ],
        )

        all_actors = self._scene.get_all_actors()
        target_object_name = self._target_object_name
        for articulations in self._scene.get_all_articulations():
            if articulations.name == self._target_object_name:
                target_object_name = [l.name for l in articulations.get_links()]
            all_actors += articulations.get_links()
        self.objects = []
        self.target_object = []
        for actor in all_actors:
            if actor.name in self.robot.link_names:
                continue
            elif actor.name in target_object_name:
                self.target_object.append(actor)
            else:
                self.objects.append(actor)

        self._step_num = 0
        self._n_episode += 1
        self._episodic_reward = 0
        self._episodic_oracle_reward = 0
        self._episode_dist2tgt = 0
        self._episode_dist2goal = 0
        self._maniskill2_reward = 0
        self._maniskill2_info = {}
        self._views = {"front": [], "wrist": []}

        obs = self.get_observation()
        return obs["original_obs"], {}

    def update_obs_space_from_scene(self):
        scene_observation_space = {
            "target_object_pose": spaces.Box(low=-10, high=10, shape=(6,)),
            "goal_pose": spaces.Box(low=-10, high=10, shape=(6,)),
            "object_poses": spaces.Box(low=-10, high=10, shape=(6,)),
            "distance_to_target": spaces.Box(low=-10, high=10, shape=(3,)),
            "distance_to_init": spaces.Box(low=-10, high=10, shape=(6,)),
            "distance_to_goal": spaces.Box(low=-10, high=10, shape=(3,)),
            "contacted": spaces.Box(low=-0, high=1, shape=(2,)),
            "collided": spaces.Box(low=-0, high=1, shape=(2,)),
        }
        self.observation_space.update(scene_observation_space)

    def update_obs_space_from_robot(self):
        robot_observation_space = {
            "joint_positions": self._env.observation_space["agent"]["qpos"],
            "joint_velocities": self._env.observation_space["agent"]["qvel"],
            "joint_reaction_forces": spaces.Box(
                low=-1000,
                high=100,
                shape=self._env.observation_space["agent"]["qpos"].shape,
            ),
            "ee_pose": spaces.Box(low=-10, high=10, shape=(6,)),
            "ee_reaction_force": spaces.Box(low=-1000, high=1000, shape=(3,)),
        }
        self.observation_space.update(robot_observation_space)

    def update_obs_space_from_camera(self):
        color_dim = [0, 0, 3]
        depth_dim = [0, 0, 3]
        for k in self._env.observation_space["image"]:
            tmp_color_shape = self._env.observation_space["image"][k][
                "Color"
            ].shape
            color_dim[0] += tmp_color_shape[0]
            color_dim[1] = tmp_color_shape[1]
            tmp_depth_shape = self._env.observation_space["image"][k][
                "Position"
            ].shape
            depth_dim[0] += tmp_depth_shape[0]
            depth_dim[1] += tmp_depth_shape[1]
        for cam_name in ["wrist", "front"]:
            self.observation_space.update(
                {
                    f"{cam_name}_rgb": spaces.Box(
                        low=0, high=255, shape=tuple(color_dim)
                    ),
                    f"{cam_name}_depth": spaces.Box(
                        low=0, high=65.535, shape=tuple(depth_dim)
                    ),
                }
            )

    @property
    def render_image(self):
        return self._render_image

    def set_render_image(self, enable_render: bool):
        self._render_image = enable_render

    def get_observation(self, obs=None):
        if obs is None:
            obs = self._env.get_obs()
        state_obs = flatten_state_dict(obs)
        # robot observation
        observation = {
            "original_obs": state_obs,
            "joint_positions": obs["agent"]["qpos"],
            "joint_velocities": obs["agent"]["qvel"],
            "joint_reaction_forces": np.empty(obs["agent"]["qpos"].shape),
            "ee_pose": np.concatenate(self.robot.ee_pose),
            "ee_reaction_force": self.robot.ee_reaction_force,
            "action": self.action_space.sample().fill(0),
        }

        # camera observation
        if self._obs_mode == "image":
            rgbs = [
                (obs["image"][k]["Color"][..., :3] * 255).astype(np.uint8)
                for k in obs["image"]
            ]
            rgbs = np.concatenate(rgbs, 0)
            depths = [
                (obs["image"][k]["Position"][..., 2] * 1000)
                .astype(np.uint16)[..., np.newaxis]
                .repeat(3, axis=-1)
                for k in obs["image"]
            ]
            depths = np.concatenate(depths, 0)
            observation.update({"front_rgb": rgbs, "front_depth": depths})
            observation.update({"wrist_rgb": rgbs, "wrist_depth": depths})
            self._views["front"].append(rgbs)
            self._views["wrist"].append(rgbs)
        elif self._render_image:
            while True:
                try:
                    rgb = self._env.render()
                    break
                except:
                    print("Rendering error, Rerender in 1s...")
                    time.sleep(1.0)
            for cam_name in self._views:
                self._views[cam_name].append(rgb)

        # scene observation
        observation["object_poses"] = np.empty(6)
        if self._target_pose_key is not None:
            observation["target_object_pose"] = obs["extra"][
                self._target_pose_key
            ]
        else:
            observation["target_object_pose"] = np.empty(6)
        if self._distance_to_target_key is None:
            observation["distance_to_target"] = (
                observation["ee_pose"][:3] - observation["target_object_pose"]
            )
        else:
            observation["distance_to_target"] = obs["extra"][
                self._distance_to_target_key
            ]

        if self._distance_to_init_key is not None:
            observation["distance_to_init"] = obs["extra"][
                self._distance_to_init_key
            ]
        else:
            observation["distance_to_init"] = np.empty(6)

        if self._goal_pose_key is not None:
            observation["goal_pose"] = obs["extra"][self._goal_pose_key]
        else:
            observation["goal_pose"] = np.empty(6)

        if self._distance_to_goal_key is not None:
            observation["distance_to_goal"] = obs["extra"][
                self._distance_to_goal_key
            ]
        else:
            observation["distance_to_goal"] = np.empty((3,))

        if self.robot.gripper_in_contact(self.target_object):
            observation["contacted"] = np.array([1.0, 0.0])
        else:
            observation["contacted"] = np.array([0.0, 1.0])

        if self.detect_collision():
            observation["collided"] = np.array([1.0, 0.0])
        else:
            observation["collided"] = np.array([0.0, 1.0])

        # filter relevant observations only
        relevant_obs = {
            k: v
            for k, v in observation.items()
            if k in self._relevant_obs_names
        }
        return relevant_obs

    @property
    def camera_views(self):
        return {name: self._views[name][-1] for name in self._views}

    def step(self, action: np.ndarray, log=False):
        (
            next_obs,
            self._maniskill2_reward,
            done,
            _,
            self._maniskill2_info,
        ) = self._env.step(action)
        if log:
            print("    Reached pose:", self.robot.ee_pose[0])

        self._step_num += 1
        next_obs = self.get_observation()
        next_obs["action"] = action
        reward, reward_vars = self.get_reward(next_obs)
        done = self.get_termination(done)

        self._episodic_reward += reward
        self._episodic_oracle_reward += self._maniskill2_reward
        info = {"success": self._maniskill2_info["success"]}
        truncated = self._step_num >= self.max_steps
        obs_vars = deepcopy(next_obs)
        obs_vars.pop("original_obs")
        info["obs_vars"] = obs_vars
        info["reward_vars"] = reward_vars
        if done:
            self._episode_dist2tgt += np.linalg.norm(
                next_obs["distance_to_target"]
            )
            self._episode_dist2goal += np.linalg.norm(
                next_obs["distance_to_goal"]
            )
            if len(self._views["front"]) > 0:
                with open(
                    f"{self._ckpt_dir}/{uuid.uuid3(uuid.NAMESPACE_DNS, str(self._episodic_reward))}.npy",
                    "wb",
                ) as fp:
                    np.save(
                        fp,
                        np.stack(self._views["front"]).transpose([0, 3, 1, 2]),
                    )
            info.update(
                {
                    "episodic_reward": self._episodic_reward,
                    "episode_length": self._step_num,
                    "episode_dist2tgt": self._episode_dist2tgt,
                    "episode_dist2goal": self._episode_dist2goal,
                    "oracle_reward": self._episodic_oracle_reward,
                }
            )

        return next_obs["original_obs"], reward, done, truncated, info

    def set_terminal_at_success(self, terminal_at_success: bool):
        self._terminal_at_success = terminal_at_success

    def get_termination(self, done) -> bool:
        if self._terminal_at_success and (
            done or (self._step_num >= self.max_steps)
        ):
            return True
        elif (not self._terminal_at_success) and (
            self._step_num >= self.max_steps
        ):
            return True
        else:
            return False

    def get_maniskill2_reward(self, obs) -> float:
        return self._maniskill2_reward, self._maniskill2_info

    def setup_reward_func(self, reward_func: str) -> None:
        # if function being passed
        if "def " in reward_func:
            llm_reward_func = reward_func.replace("(obs,", "(self, obs,")
            llm_reward_func = llm_reward_func.replace(
                "ee_reaction_forece", "ee_reaction_force"
            )
            llm_reward_func = llm_reward_func.replace("env.", "self.")
            llm_reward_func = llm_reward_func.replace(
                "obj.in_contact(self.robot)",
                "self.robot.gripper_in_contact(obj)",
            )
            llm_reward_func = llm_reward_func.replace(
                "return total_reward",
                "vars = {}\n    for k, v in locals().items():\n        if type(v) in [np.int, np.float32, np.float64, int, float, bool, np.bool_]:\n            vars[k]=v\n    return total_reward, vars\n",
            )
            exec(llm_reward_func)
            self.get_reward = types.MethodType(eval("get_reward"), self)
        # if existing function name being passed
        else:
            self.get_reward = getattr(self, reward_func)

    def detect_collision(self, ignore_target=True):
        force_magnitude = np.linalg.norm(self.robot.ee_reaction_force[:3])
        if force_magnitude <= 0:
            return False
        elif self.robot.gripper_in_contact(self.objects):
            return True
        elif self.robot.gripper_in_contact(self.target_object):
            if ignore_target:
                return False
            else:
                return True
        else:
            return False

    def detect_success(self, obs):
        return self._maniskill2_info["success"]


class ManiSkill2DualArmMobileBaseTaskWrapper(gym.Wrapper):
    def __init__(
        self,
        env_name: str,
        target_object_name: str,
        target_pose_key: str,
        goal_pose_key: str,
        distance_to_target_key: str,
        distance_to_goal_key: str,
        distance_to_init_key: str,
        obs_mode: str = "state_dict",
        reward_mode: str = "dense",
        terminal_at_success: bool = True,
        control_mode: str = "base_pd_joint_vel_arm_pd_ee_delta_pose",
        max_steps: int = 25,
        relevant_obs_names: List[str] = None,
        reward_func: Callable = None,
        render_image: bool = False,
        ckpt_dir: str = "",
    ) -> None:
        self._render_image = render_image
        self._obs_mode = obs_mode
        self._env = gym.make(
            env_name,
            obs_mode=self._obs_mode,
            reward_mode=reward_mode,
            control_mode=control_mode,
            render_mode="cameras",
            max_episode_steps=max_steps,
        )
        super().__init__(self._env)
        self._target_object_name = target_object_name

        self._relevant_obs_names = relevant_obs_names
        self._target_pose_key = target_pose_key
        self._goal_pose_key = goal_pose_key
        self._distance_to_target_key = distance_to_target_key
        self._distance_to_goal_key = distance_to_goal_key
        self._distance_to_init_key = distance_to_init_key
        assert (
            "joint_reaction_forces" not in self._relevant_obs_names
        ), "Unsupported observation joint_reaction_forces."
        assert (
            "object_poses" not in self._relevant_obs_names
        ), "Unsupported observation object_poses."

        obs, info = self._env.reset(seed=2022)
        state_obs = flatten_state_dict(obs)
        if False:
            self.observation_space = {
                "original_obs": spaces.Box(
                    low=-np.inf, high=np.inf, shape=state_obs.shape
                ),
                "action": self._env.action.space,
            }
            self.update_obs_space_from_scene()
            self.update_obs_space_from_robot()
            if self._obs_mode == "image":
                self.update_obs_space_from_camera()
            if self._relevant_obs_names is None:
                self._relevant_obs_names = list(self.observation_space.keys())
            self.observation_space = spaces.Dict(
                {
                    k: v
                    for k, v in self.observation_space.items()
                    if k in self._relevant_obs_names
                }
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=state_obs.shape
            )

        self.action_space = self._env.action_space

        self._terminal_at_success = terminal_at_success
        if reward_func is not None:
            self.setup_reward_func(reward_func)
        else:
            self.get_reward = self.get_maniskill2_reward

        self._ckpt_dir = ckpt_dir
        self._n_episode = -1
        self.max_steps = max_steps  # self._env.max_episode_steps
        _ = self.reset()

    @property
    def sapien(self):
        return self._env

    def get_world_pcd(self, obj, pcd):
        return transform_points(obj.pose.to_transformation_matrix(), pcd)

    def reset(self, seed=None, options=None):
        self._env.reset(seed=seed, options=options)
        self._scene = self._env.agent.scene
        self.robot = ManiSkill2DualArmRobotWrapper(self._env)

        all_actors = self._scene.get_all_actors()
        target_object_name = self._target_object_name
        for articulations in self._scene.get_all_articulations():
            if articulations.name == self._target_object_name:
                target_object_name = [l.name for l in articulations.get_links()]
            all_actors += articulations.get_links()
        self.objects = []
        self.target_object = []
        for actor in all_actors:
            if actor.name in self.robot.link_names:
                continue
            elif actor.name in target_object_name:
                self.target_object.append(actor)
            else:
                self.objects.append(actor)

        self._step_num = 0
        self._n_episode += 1
        self._episodic_reward = 0
        self._episodic_oracle_reward = 0
        self._episode_dist2tgt = 0
        self._episode_dist2goal = 0
        self._maniskill2_reward = 0
        self._maniskill2_info = {}
        self._views = {"front": [], "wrist": []}

        obs = self.get_observation()
        return obs["original_obs"], {}

    def update_obs_space_from_scene(self):
        scene_observation_space = {
            "target_object_pose": spaces.Box(low=-10, high=10, shape=(6,)),
            "goal_pose": spaces.Box(low=-10, high=10, shape=(6,)),
            "object_poses": spaces.Box(low=-10, high=10, shape=(6,)),
            "distance_to_target": spaces.Box(low=-10, high=10, shape=(3,)),
            "distance_to_init": spaces.Box(low=-10, high=10, shape=(6,)),
            "distance_to_goal": spaces.Box(low=-10, high=10, shape=(3,)),
            "contacted": spaces.Box(low=-0, high=1, shape=(4,)),
            "collided": spaces.Box(low=-0, high=1, shape=(4,)),
        }
        self.observation_space.update(scene_observation_space)

    def update_obs_space_from_robot(self):
        robot_observation_space = {
            "joint_positions": self._env.observation_space["agent"]["qpos"],
            "joint_velocities": self._env.observation_space["agent"]["qvel"],
            "joint_reaction_forces": spaces.Box(
                low=-1000,
                high=100,
                shape=self._env.observation_space["agent"]["qpos"].shape,
            ),
            "ee_pose": spaces.Box(low=-10, high=10, shape=(12,)),
            "ee_reaction_force": spaces.Box(low=-1000, high=1000, shape=(6,)),
        }
        self.observation_space.update(robot_observation_space)

    def update_obs_space_from_camera(self):
        color_dim = [0, 0, 3]
        depth_dim = [0, 0, 3]
        for k in self._env.observation_space["image"]:
            tmp_color_shape = self._env.observation_space["image"][k][
                "Color"
            ].shape
            color_dim[0] += tmp_color_shape[0]
            color_dim[1] = tmp_color_shape[1]
            tmp_depth_shape = self._env.observation_space["image"][k][
                "Position"
            ].shape
            depth_dim[0] += tmp_depth_shape[0]
            depth_dim[1] += tmp_depth_shape[1]
        for cam_name in ["wrist", "front"]:
            self.observation_space.update(
                {
                    f"{cam_name}_rgb": spaces.Box(
                        low=0, high=255, shape=tuple(color_dim)
                    ),
                    f"{cam_name}_depth": spaces.Box(
                        low=0, high=65.535, shape=tuple(depth_dim)
                    ),
                }
            )

    @property
    def render_image(self):
        return self._render_image

    def set_render_image(self, enable_render: bool):
        self._render_image = enable_render

    def get_observation(self, obs=None):
        if obs is None:
            obs = self._env.get_obs()
        state_obs = flatten_state_dict(obs)
        # robot observation
        observation = {
            "original_obs": state_obs,
            "joint_positions": obs["agent"]["qpos"],
            "joint_velocities": obs["agent"]["qvel"],
            "joint_reaction_forces": np.empty(obs["agent"]["qpos"].shape),
            "ee_pose": np.concatenate(self.robot.ee_pose),
            "ee_reaction_force": self.robot.ee_reaction_force,
            "action": self.action_space.sample().fill(0),
        }

        # camera observation
        if self._obs_mode == "image":
            rgbs = [
                (obs["image"][k]["Color"][..., :3] * 255).astype(np.uint8)
                for k in obs["image"]
            ]
            rgbs = np.concatenate(rgbs, 0)
            depths = [
                (obs["image"][k]["Position"][..., 2] * 1000)
                .astype(np.uint16)[..., np.newaxis]
                .repeat(3, axis=-1)
                for k in obs["image"]
            ]
            depths = np.concatenate(depths, 0)
            observation.update({"front_rgb": rgbs, "front_depth": depths})
            observation.update({"wrist_rgb": rgbs, "wrist_depth": depths})
            self._views["front"].append(rgbs)
            self._views["wrist"].append(rgbs)
        elif self._render_image:
            while True:
                try:
                    rgb = self._env.render()
                    break
                except:
                    print("Rendering error, Rerender in 1s...")
                    time.sleep(1.0)
            for cam_name in self._views:
                self._views[cam_name].append(rgb)

        # scene observation
        observation["object_poses"] = np.empty(6)
        if self._target_pose_key is not None:
            observation["target_object_pose"] = obs["extra"][
                self._target_pose_key
            ]
        else:
            observation["target_object_pose"] = np.empty(6)
        if self._distance_to_target_key is None:
            observation["distance_to_target"] = observation["ee_pose"][
                :6
            ] - np.concatenate([observation["target_object_pose"]] * 2)
        else:
            observation["distance_to_target"] = obs["extra"][
                self._distance_to_target_key
            ]

        if self._distance_to_init_key is not None:
            observation["distance_to_init"] = obs["extra"][
                self._distance_to_init_key
            ]
        else:
            observation["distance_to_init"] = np.empty(6)

        if self._goal_pose_key is not None:
            observation["goal_pose"] = obs["extra"][self._goal_pose_key]
        else:
            observation["goal_pose"] = np.empty(6)

        if self._distance_to_goal_key is not None:
            observation["distance_to_goal"] = obs["extra"][
                self._distance_to_goal_key
            ]
        else:
            observation["distance_to_goal"] = np.empty((3,))

        contact_embed = {True: [1.0, 0.0], False: [0.0, 1.0]}
        left_contact, right_contact = self.robot.gripper_in_contact(
            self.target_object, any=False
        )
        observation["contacted"] = np.concatenate(
            [contact_embed[left_contact], contact_embed[right_contact]]
        )

        collide_embed = {True: [1.0, 0.0], False: [0.0, 1.0]}
        left_collided, right_collided = self.detect_collision(any=False)
        observation["collided"] = np.concatenate(
            [collide_embed[left_collided], collide_embed[right_collided]]
        )

        # filter relevant observations only
        relevant_obs = {
            k: v
            for k, v in observation.items()
            if k in self._relevant_obs_names
        }
        return relevant_obs

    @property
    def camera_views(self):
        return {name: self._views[name][-1] for name in self._views}

    def step(self, action: np.ndarray, log=False):
        (
            next_obs,
            self._maniskill2_reward,
            done,
            truncated,
            self._maniskill2_info,
        ) = self._env.step(action)
        if log:
            print("    Reached pose:", self.robot.ee_pose[0])

        self._step_num += 1
        next_obs = self.get_observation()
        next_obs["action"] = action
        reward, reward_vars = self.get_reward(next_obs)
        done = self.get_termination(done)

        self._episodic_reward += reward
        self._episodic_oracle_reward += self._maniskill2_reward
        info = {"success": self._maniskill2_info["success"]}
        truncated = self._step_num >= self.max_steps
        obs_vars = deepcopy(next_obs)
        obs_vars.pop("original_obs")
        info["obs_vars"] = obs_vars
        info["reward_vars"] = reward_vars
        if done:
            self._episode_dist2tgt += np.linalg.norm(
                next_obs["distance_to_target"]
            )
            self._episode_dist2goal += np.linalg.norm(
                next_obs["distance_to_goal"]
            )
            if len(self._views["front"]) > 0:
                with open(
                    f"{self._ckpt_dir}/{uuid.uuid3(uuid.NAMESPACE_DNS, str(self._episodic_reward))}.npy",
                    "wb",
                ) as fp:
                    np.save(
                        fp,
                        np.stack(self._views["front"]).transpose([0, 3, 1, 2]),
                    )
            info.update(
                {
                    "episodic_reward": self._episodic_reward,
                    "episode_length": self._step_num,
                    "episode_dist2tgt": self._episode_dist2tgt,
                    "episode_dist2goal": self._episode_dist2goal,
                    "oracle_reward": self._episodic_oracle_reward,
                }
            )

        return next_obs["original_obs"], reward, done, truncated, info

    def set_terminal_at_success(self, terminal_at_success: bool):
        self._terminal_at_success = terminal_at_success

    def get_termination(self, done) -> bool:
        if self._terminal_at_success and (
            done or (self._step_num >= self.max_steps)
        ):
            return True
        elif (not self._terminal_at_success) and (
            self._step_num >= self.max_steps
        ):
            return True
        else:
            return False

    def get_maniskill2_reward(self, obs) -> float:
        return self._maniskill2_reward, self._maniskill2_info

    def setup_reward_func(self, reward_func: str) -> None:
        # if function being passed
        if "def " in reward_func:
            llm_reward_func = reward_func.replace("(obs,", "(self, obs,")
            llm_reward_func = llm_reward_func.replace(
                "ee_reaction_forece", "ee_reaction_force"
            )
            llm_reward_func = llm_reward_func.replace("env.", "self.")
            llm_reward_func = llm_reward_func.replace(
                "obj.in_contact(self.robot)",
                "self.robot.gripper_in_contact(obj)",
            )
            llm_reward_func = llm_reward_func.replace(
                "return total_reward",
                "vars = {}\n    for k, v in locals().items():\n        if type(v) in [np.int, np.float32, np.float64, int, float, bool, np.bool_]:\n            vars[k]=v\n    return total_reward, vars\n",
            )
            exec(llm_reward_func)
            self.get_reward = types.MethodType(eval("get_reward"), self)
        # if existing function name being passed
        else:
            self.get_reward = getattr(self, reward_func)

    def detect_collision(self, ignore_target=True, any=True):
        collided = []
        for robot in [self.robot.left_arm, self.robot.right_arm]:
            force_magnitude = np.linalg.norm(robot.ee_reaction_force[:3])
            if force_magnitude <= 0:
                collided.append(False)
            elif robot.gripper_in_contact(self.objects):
                collided.append(True)
            elif robot.gripper_in_contact(self.target_object):
                if ignore_target:
                    collided.append(False)
                else:
                    collided.append(True)
            else:
                # collided.append(True)
                collided.append(False)
        left_collided, right_collided = collided
        if any:
            return left_collided or right_collided
        else:
            return left_collided, right_collided

    def detect_success(self, obs):
        return self._maniskill2_info["success"]
