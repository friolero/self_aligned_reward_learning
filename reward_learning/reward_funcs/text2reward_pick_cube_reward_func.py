# (zero-shot) Instruction: Pick up cube A and move it to the 3D goal position.
llm_reward_func = """
def get_reward(obs, weight_dist_gripper_cube=0.3, weight_dist_cube_goal=0.5, weight_grasping_cube=0.2, weight_action_reg=0.01):

    action_magnitude = (np.clip(obs["action"], -1, 1)**2).sum()
    action_reg_reward = -weight_action_reg * action_magnitude

    # Calculate distance between gripper and cube
    gripper_pos = env.sapien.tcp.pose.p
    cube_pos = env.sapien.obj.pose.p
    dist_gripper_cube = np.linalg.norm(gripper_pos - cube_pos)

    # Calculate distance between cube and goal
    goal_pos = env.sapien.goal_pos
    dist_cube_goal = np.linalg.norm(goal_pos - cube_pos)

    # Check if the robot is grasping the cube
    grasping_cube = env.sapien.agent.check_grasp(env.sapien.obj)

    # Define reward components
    dist_gripper_cube_reward = -1.0 * dist_gripper_cube
    dist_cube_goal_reward = -1.0 * dist_cube_goal
    grasping_cube_reward = 1.0 if grasping_cube else -1.0

    # Calculate total reward
    total_reward = weight_dist_gripper_cube * dist_gripper_cube_reward + weight_dist_cube_goal * dist_cube_goal_reward + weight_grasping_cube * grasping_cube_reward + weight_action_reg * action_reg_reward

    return total_reward
"""
