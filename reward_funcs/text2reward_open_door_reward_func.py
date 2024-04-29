llm_reward_func = """
def get_reward(obs, w_distance=0.4, w_goal=0.4, w_action=0.2, static_reward_value=1.0):

    # Calculate distance between robot's gripper and the cabinet handle
    handle_pcd = env.get_world_pcd(env.sapien.target_link, env.sapien.target_handle_pcd)
    ee_cords = env.sapien.agent.get_ee_coords_sample().reshape(-1, 3)
    distance = cdist(ee_cords, handle_pcd).min()
    distance_reward = -w_distance * distance  # Negative reward since we want to minimize the distance

    # Calculate the difference between current state of cabinet drawer and its goal state
    # Positive reward since we want to maximize the qpos
    goal_diff = env.sapien.link_qpos - env.sapien.target_qpos
    goal_reward = w_goal * goal_diff

    # Add regularization of robot's action, penalize large actions
    action_magnitude = np.linalg.norm(obs["action"])
    action_reward = -w_action * action_magnitude

    # Check if the target drawer is static, if so, give a large positive reward
    target_link_static = env.sapien.check_actor_static(env.sapien.target_link, max_v=0.1, max_ang_v=1)
    static_reward = static_reward_value if target_link_static else 0.0

    # Combine different parts of the reward
    total_reward = distance_reward + goal_reward + action_reward + static_reward

    return total_reward
"""
