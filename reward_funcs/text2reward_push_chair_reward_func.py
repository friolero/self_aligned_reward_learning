llm_reward_func = """
def get_reward(obs, w_dist=0.2, w_target=0.2, w_vel=0.3, w_tilt=0.2, w_reg=0.1):

    # Stage 1: Encourage the robot to move towards the chair
    # Get the distance between the robot's gripper and the chair
    gripper_coords = env.sapien.agent.get_ee_coords()
    chair_pcd = env.get_world_pcd(env.sapien.chair, env.sapien.chair_pcd)
    dist_to_chair = cdist(gripper_coords, chair_pcd).min(-1).mean()

    # Get the difference between the chair's current and target position
    chair_to_target_dist = np.linalg.norm(env.sapien.chair.pose.p[:2] - env.sapien.target_xy)

    # The smaller the distance, the larger the reward
    dist_reward = -dist_to_chair
    # The closer the chair is to the target, the larger the reward
    target_reward = -chair_to_target_dist

    # Stage 2: Encourage the robot to push the chair towards the target location
    # Get the velocity of the chair
    chair_vel = env.sapien.root_link.velocity[:2]
    chair_vel_alignment = np.dot(chair_vel, (env.sapien.target_xy - env.sapien.chair.pose.p[:2])) / (np.linalg.norm(chair_vel) * chair_to_target_dist)
    # The faster the chair moves towards the target, the larger the reward
    vel_reward = chair_vel_alignment

    # Stage 3: Prevent the chair from falling over
    # Calculate the tilt angle of the chair
    z_axis_chair = env.sapien.root_link.pose.to_transformation_matrix()[:3, 2]
    chair_tilt = np.arccos(z_axis_chair[2])
    # The smaller the tilt, the larger the reward
    tilt_reward = -chair_tilt

    # Regularization of the robot's action
    action_magnitude = np.square(obs["action"]).sum()
    action_reg_reward = -action_magnitude

    # Final reward
    total_reward = w_dist * dist_reward + w_target * target_reward + w_vel * vel_reward + w_tilt * tilt_reward + w_reg * action_reg_reward

    return total_reward
"""
