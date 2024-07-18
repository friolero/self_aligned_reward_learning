llm_reward_func = """
def get_reward(obs, approach_distance_weight=1.0, grasp_reward_value=10.0, goal_distance_weight=1.0, collision_penalty_value=-10.0, maintaining_position_reward_value=5.0):
    # Extract features from observations
    distance_to_target = np.linalg.norm(obs['distance_to_target'])
    distance_to_goal = np.linalg.norm(obs['distance_to_goal'])

    # Substep 1: Approaching - Reward is inversely proportional to the distance to the target
    approach_reward = - approach_distance_weight * distance_to_target

    # Substep 2: Grasping - Assign a reward if the object is grasped
    grasped = env.robot.is_grasping(env.target_object)
    grasp_reward = grasp_reward_value if grasped else 0

    # Substep 3: Transporting - Reward is inversely proportional to the distance to the goal
    transport_reward = - goal_distance_weight * distance_to_goal

    # Substep 4: Maintaining Position - Assign a reward for minimal distance to goal
    maintain_reward = maintaining_position_reward_value if distance_to_goal < 0.05 else 0

    # Apply a penalty if there is a collision
    collision_detected = env.detect_collision()
    collision_penalty = collision_penalty_value if collision_detected else 0

    total_reward = approach_reward + grasp_reward + transport_reward + maintain_reward + collision_penalty

    return total_reward
"""
