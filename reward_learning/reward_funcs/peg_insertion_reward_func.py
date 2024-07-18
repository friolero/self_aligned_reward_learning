llm_reward_func = """
def get_reward(obs, distance_to_peg_weight=1.0, alignment_ee_to_peg_weight=2.0, grasp_reward_value=5.0, distance_to_hole_weight=1.0, alignment_peg_to_hole_weight=2.0, collision_penalty_value=-10.0, force_threshold=5.0):

    # 1. Reward for reducing distance to the peg
    distance_to_peg = np.linalg.norm(obs['distance_to_target'])
    approach_reward = distance_to_peg_weight * (- distance_to_peg)

    # 2. Reward for aligning with the peg
    alignment_error_ee_to_peg = np.linalg.norm(env.robot.check_ee_alignment(env.target_object)) / 10
    ee_to_peg_alignment_reward = alignment_ee_to_peg_weight * (- alignment_error_ee_to_peg)

    # 3. Reward for grasping the peg
    grasped = env.robot.is_grasping(env.target_object)
    grasp_reward = grasp_reward_value if grasped else 0

    lift_and_move_reward = 0
    peg_to_hole_alignment_reward = 0
    if grasped:
        # 4. Reward for moving towards the hole
        # Inverse of the distance between target object pose and goal pose
        peg_to_hole_distance = np.linalg.norm(obs["distance_to_goal"])
        lift_and_move_reward = distance_to_hole_weight * (- peg_to_hole_distance)

        # 5. Reward for aligning peg with hole
        # Similar to step 2 but considering target object pose and goal pose
        alignment_error_peg_to_hole = np.linalg.norm(obs['target_object_pose'][3:] - obs['goal_pose'][3:]) / 10
        peg_to_hole_alignment_reward = alignment_peg_to_hole_weight * (- alignment_error_peg_to_hole)

    # Penalty for collision
    collision_detected = env.detect_collision()
    collision_penalty = collision_penalty_value if collision_detected else 0

    total_reward = approach_reward + ee_to_peg_alignment_reward + grasp_reward + lift_and_move_reward + peg_to_hole_alignment_reward + collision_penalty

    return total_reward
"""
