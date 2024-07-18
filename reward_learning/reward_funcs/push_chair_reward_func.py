llm_reward_func = """
def get_reward(obs, approach_weight=1.0, movement_weight=1.0, collision_penalty_weight=1.0):

    # Reward for minimizing the distance between the robot gripper and the chair
    gripper_to_chair_dist = np.linalg.norm(obs['distance_to_target'])
    approach_reward = -gripper_to_chair_dist  # Negative value: smaller distance is better

    # Reward for moving the chair towards the target position
    # Assuming the target position is part of the environment's state
    chair_to_target_dist = np.linalg.norm(obs['distance_to_goal'][:2])
    movement_reward = -chair_to_target_dist  # Negative value: smaller distance is better

    # Penalty for collisions
    collision_detected = env.detect_collision()
    collision_penalty = -1.0 if collision_detected else 0.0

    # Calculate total reward
    total_reward = approach_weight * approach_reward + movement_weight * movement_reward + collision_penalty_weight * collision_penalty

    return total_reward
"""
