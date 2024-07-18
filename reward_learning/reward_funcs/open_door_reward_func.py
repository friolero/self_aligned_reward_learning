llm_reward_func = """
def get_reward(obs, alignment_weight=1.0, approach_weight=1.0, grasp_weight=1.0, pull_weight=1.0, collision_penalty_weight=1.0, non_progress_penalty_weight=1.0, distance_penalty_weight=0.1):

    # Extract relevant features from observations
    distance_to_handle = np.linalg.norm(obs['distance_to_target'])
    distance_to_goal = obs['distance_to_goal'][0]

    # Check for contact and collision
    contacted = env.robot.gripper_in_contact(env.target_object)
    collision_detected = env.detect_collision()

    alignment_reward = - alignment_weight * distance_to_handle

    # Approach reward: Encouraging the EE to get closer to the handle
    # Higher when the distance is smaller, so we take the negative distance
    approach_reward = - approach_weight * distance_to_handle

    # Grasp reward: Binary reward for making contact with the handle
    grasp_reward = grasp_weight * 1.0 if contacted else 0.0

    # Pull reward: Encouraging the EE to get the handle closer to the goal position
    # Higher when the distance is smaller, so we take the negative distance
    pull_reward = -pull_weight * distance_to_goal

    # Collision penalty: Penalizing any collision detected during interaction
    collision_penalty = -collision_penalty_weight * 1.0 if collision_detected else 0.0

    # Non-progress penalty: If there's no contact and no reduction in distance to the handle,
    # penalize to encourage the EE to move towards the handle
    non_progress_penalty = -non_progress_penalty_weight * 1.0 if not contacted and distance_to_handle > 0.02 else 0.0  # Threshold of 2cm

    # Excessive distance penalty: Discouraging the EE from being too far from the handle
    excessive_distance_penalty = - distance_penalty_weight * distance_to_handle if distance_to_handle > 0.15 else 0.0  # Threshold of 15cm

    # Combine rewards and penalties into total reward
    total_reward = alignment_reward + approach_reward + grasp_reward + pull_reward + collision_penalty + non_progress_penalty + excessive_distance_penalty

    return total_reward
"""
