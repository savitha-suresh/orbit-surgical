import torch
import numpy as np
from isaaclab.envs import DirectMARLEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import sample_uniform, quat_from_angle_axis, quat_mul, saturate

from isaaclab.utils.math import subtract_frame_transforms

from .joint_pos_env_cfg import BlockHandoverEnvCfg
from .phase_detector import Phases, PhaseDetector




class DualArmHandoverEnv(DirectMARLEnv):
    cfg: BlockHandoverEnvCfg

    def __init__(self, cfg: BlockHandoverEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.phase_detector = PhaseDetector(cfg)
        self.ee_link_name = self.cfg.ee_link_name
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor([0.0, 0.0, 0.1], device=self.device)

        self.r1_init_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.r1_init_pos[:, :] = torch.tensor([0.18, 0.0, 0.15], device=self.device)
        self.r2_init_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.r2_init_pos[:, :] = torch.tensor([-0.18, 0.0, 0.15], device=self.device)
        self.current_phases = torch.zeros((self.num_envs, len(Phases)), dtype=torch.float, device=self.device)
        self.current_phases[:, Phases.REACH_OBJ.value] = 1.0



        self.num_hand_dofs = self.robot_1.num_joints
        self.actuated_dof_indices = []
        for joint_name in self.robot_1.joint_names:
            self.actuated_dof_indices.append(self.robot_1.joint_names.index(joint_name))

        # buffers for position targets
        self.robot_1_dof_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.robot_1_prev_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.robot_1_curr_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.robot_2_dof_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.robot_2_prev_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.robot_2_curr_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )



        joint_pos_limits = self.robot_1.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]


    
        

    def _setup_scene(self):
        self.robot_1 = Articulation(self.cfg.scene.robot_1)
        self.robot_2 = Articulation(self.cfg.scene.robot_2)
        self.object = RigidObject(self.cfg.scene.object)

        self.scene.clone_environments(copy_from_source=False)

        self.scene.articulations["robot_1"] = self.robot_1
        self.scene.articulations["robot_2"] = self.robot_2
        self.scene.rigid_objects["object"] = self.object

        
     
    def _pre_physics_step(self, actions):
    
        self.actions = actions
        

    def _compute_intermediate_values(self):
       

        # data for object
        self.object_pos = self.object.data.root_pos_w
        self.object_rot = self.object.data.root_quat_w
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

    def _get_ee_position(self, robot):
        ee_pos = robot.data.body_pos_w[:, robot.find_bodies(self.cfg.ee_link_name)[0]]
        return ee_pos.squeeze(1)

    def _get_ee_pose(self, robot):

        # Get robot 1 end-effector pose
        
        ee_pos = robot.data.body_pos_w[:, robot.find_bodies(self.cfg.ee_link_name)[0]]
        ee_quat = robot.data.body_quat_w[:, robot.find_bodies(self.cfg.ee_link_name)[0]]
        ee_pose = torch.cat([ee_pos, ee_quat], dim=-1)  # [num_envs, 7]
        ee_pose = ee_pose.squeeze(1)
        return ee_pose

    def _get_observations(self):
           
        
        observations = {}
        
        
        #print(self.current_phases)
        # Process each robot separately
        for robot_name in self.cfg.possible_agents:
            robot = self.scene.articulations[robot_name]
 
            # Resolve joint IDs
            joint_ids = [robot.joint_names.index(name) for name in robot.joint_names]
  
            joint_pos_rel = robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]
             
            joint_vel_rel = robot.data.joint_vel[:, joint_ids] - robot.data.default_joint_vel[:, joint_ids]
            
            # 3. Object position in robot root frame
           
            # # 4. Target object position from command generator
            # # You'll need to implement this based on your command generation logic
            # if hasattr(self, 'command_manager') and self.command_manager is not None:
            #     # If you have a command manager
            #     target_commands = self.command_manager.get_command("object_pose")
            #     if target_commands.shape[-1] >= 3:
            #         target_object_pos = target_commands[:, :3]  # Take position part
            #     else:
            #         target_object_pos = target_commands
            # else:
            #     # Fallback: create dummy target or implement your own target logic
            #     target_object_pos = torch.zeros((self.num_envs, 3), device=self.device)
            #     # Example: You might want to set a specific target position
            #     # target_object_pos = torch.tensor([[1.0, 0.0, 0.5]], device=self.device).repeat(self.num_envs, 1)
            
            # 5. Last actions
            # # Store previous actions in your environment (you'll need to track this)
            # if hasattr(self, '_previous_actions') and robot_name in self._previous_actions:
            #     last_actions = self._previous_actions[robot_name]
            # else:
            #     # Initialize with zeros if no previous actions
            #     num_actions = getattr(self.cfg, 'num_actions', joint_pos.shape[1])  # Assume action dim equals joint dim
            #     last_actions = torch.zeros((self.num_envs, num_actions), device=self.device)

            ee_pose = self._get_ee_pose(robot)
            # Concatenate all observations for this robot
            obs_list = [
                joint_pos_rel,
                joint_vel_rel,
                self._get_obj_pos(),
                ee_pose,
                self.goal_pos,
                self.goal_rot
            ]
            
            # Concatenate along the feature dimension
            robot_obs = torch.cat(obs_list, dim=-1)
            
           
            observations[robot_name] = robot_obs
    
        return observations
    

    def get_joint_ids_for_action(self, robot_name, robot):
        phase_joint_names_map = {
            Phases.REACH_OBJ: {
                "robot_1": self.cfg.body_joint_names,
            },
            Phases.GRIP_1_OPEN: {
                "robot_1": self.cfg.finger_joint_names,
            },
            Phases.LIFT: {
                "robot_1": self.cfg.body_joint_names,
            },
            Phases.REACH_GOAL_1: {
                "robot_1": self.cfg.body_joint_names,
            },
            Phases.REACH_GOAL_2: {
                "robot_2": self.cfg.body_joint_names,
            },
            Phases.GRIP_2:  {
                "robot_2": self.cfg.finger_joint_names,
            },
            Phases.RELEASE_1:  {
                "robot_1": self.cfg.finger_joint_names,
            },
            Phases.END: {
                "robot_1": self.cfg.body_joint_names,
            },
        }

        phases = self._get_phase()
        
        env_joints = {}

        
        for env_id, phase in enumerate(phases):
            joint_ids = []
            robot_joint_map = phase_joint_names_map[phase]
            joint_names = robot_joint_map.get(robot_name)
            if not joint_names:
                return {}
            joint_ids = [robot.joint_names.index(name) for name in joint_names]
            env_joints[env_id] = joint_ids
        return env_joints
    
    def mask_actions(self, actions, env_joint_dict):
        num_envs, num_joints = actions.shape
        
        # Create a mask initialized with zeros (all joints inactive by default)
        mask = torch.zeros_like(actions, dtype=torch.bool, device=actions.device)
        
        # Set mask to True for active joints in each environment
        for env_id, joint_ids in env_joint_dict.items():
            if env_id < num_envs:  # Safety check
                mask[env_id, joint_ids] = True
        
        # Apply mask: keep original actions for active joints, set others to 0
        masked_actions = actions * mask.float()
        
        return masked_actions
    
    def _apply_action(self):

        
        
        
        
        self.robot_1_curr_targets[:, self.actuated_dof_indices] = scale(
            self.actions["robot_1"],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        # self.robot_1_curr_targets[:, self.actuated_dof_indices] = (
        #     self.cfg.act_moving_average * self.robot_1_curr_targets[:, self.actuated_dof_indices]
        #     + (1.0 - self.cfg.act_moving_average) * self.robot_1_prev_targets[:, self.actuated_dof_indices]
        # )

        # Only apply moving average if delta action is large
        diff_1 = torch.abs(self.robot_1_curr_targets - self.robot_1_prev_targets)
        diff_2 = torch.abs(self.robot_2_curr_targets - self.robot_2_prev_targets)

        # Threshold below which we consider the arm to be "still"
        STABILITY_THRESHOLD = 1e-3  # You can tune this

        mask_1 = (diff_1 > STABILITY_THRESHOLD).float()
        mask_2 = (diff_2 > STABILITY_THRESHOLD).float()

        # Blend only if significant change
        self.robot_1_curr_targets = (
            mask_1 * (
                self.cfg.act_moving_average * self.robot_1_curr_targets
                + (1.0 - self.cfg.act_moving_average) * self.robot_1_prev_targets
            ) + (1 - mask_1) * self.robot_1_prev_targets
        )

        self.robot_2_curr_targets = (
            mask_2 * (
                self.cfg.act_moving_average * self.robot_2_curr_targets
                + (1.0 - self.cfg.act_moving_average) * self.robot_2_prev_targets
            ) + (1 - mask_2) * self.robot_2_prev_targets
        )

        self.robot_1_curr_targets[:, self.actuated_dof_indices] = saturate(
            self.robot_1_curr_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        
        self.robot_2_curr_targets[:, self.actuated_dof_indices] = scale(
            self.actions["robot_2"],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.robot_2_curr_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.robot_2_curr_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.robot_2_prev_targets[:, self.actuated_dof_indices]
        )
        self.robot_2_curr_targets[:, self.actuated_dof_indices] = saturate(
            self.robot_2_curr_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        # env_joints_1 = self.get_joint_ids_for_action("robot_1",
        #                                                 self.robot_1)
        # env_joints_2 = self.get_joint_ids_for_action("robot_2", self.robot_2)

        # masked_actions_1 = self.mask_actions(self.robot_1_curr_targets, env_joints_1)
        # masked_actions_2 = self.mask_actions(self.robot_2_curr_targets, env_joints_2)


        # # save current targets
        # self.robot_1_prev_targets = masked_actions_1
        # self.robot_2_prev_targets = masked_actions_2

        # # set targets
        # self.robot_1.set_joint_position_target(
        #     masked_actions_1
        # )
        # self.robot_2.set_joint_position_target(
        #     masked_actions_2
        # )

        self.robot_1.set_joint_position_target(
            self.robot_1_curr_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )
        # self.robot_2.set_joint_position_target(
        #     self.robot_2_curr_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        # )
        
    
    
    def _get_states(self):
        
        # same as observation but both agents in one vector
        obs = self._get_observations()
        
        return torch.cat([obs[agent] for agent in self.cfg.possible_agents], dim=-1)
    

    def _get_phase(self):
        # self.prev_phases = self.current_phases
        # self.current_phases = self.phase_detector.get_phases(
        #     agents=[self.robot_1, self.robot_2],
        #     obj_position=self._get_obj_pos(),
        #     batch_size=self.num_envs,
        #     goal_position=self.goal_pos,
        #     prev_phases=self.prev_phases
        # )
        # return self.current_phases
        self.current_phases = self.phase_detector.get_phases(
            agents=[self.robot_1, self.robot_2],
            obj_position=self._get_obj_pos(),
            batch_size=self.num_envs,
            goal_position=self.goal_pos,
            prev_phases = self.current_phases.clone()
        )
        return self.current_phases

    def _get_obj_pos(self):
        #return self.object.data.root_pos_w - self.scene.env_origin
        pos_all = self.object.data.root_pos_w
        pos_all[:2]+=0.001
        return  pos_all
    
    def _get_r2_stationary_rew(self, env_id):
        ee_position_2 = self._get_ee_position(self.robot_2)[env_id]
        goal_pos = self.r2_init_pos[env_id]
        dist_2 = torch.norm(goal_pos.float() - ee_position_2.float(), p=2, dim=-1)
        
        reward_2 = 2*torch.exp(-self.cfg.dist_reward_scale * dist_2)
        
        return reward_2
    
    # def _get_rewards(self):
    #     phases = self._get_phase()
        
        
    #     final_rewards = {}
    #     for robot_name in self.cfg.possible_agents:
    #         final_rewards[robot_name] = torch.zeros((self.num_envs,), dtype=torch.float, device=self.robot_1.data.device)
    #     obj_pos_all = self._get_obj_pos()
    #     # agent_name = 'robot_1'               
    #     # robot = self.scene.articulations[agent_name]
    #     # ee_position = self._get_ee_position(robot)
    #     # dist = torch.norm(obj_pos_all - ee_position, p=2, dim=-1)
        
    #     # reward = 2 * torch.exp(-self.cfg.dist_reward_scale * dist)
    #     # #print(reward, dist)
    #     # final_rewards[agent_name] = reward
        
        
    #     for env_id, phase in enumerate(phases):
    #         obj_pos = obj_pos_all[env_id]
    #         if phase == Phases.REACH_OBJ:
    #             agent_name = 'robot_1'               
    #             robot = self.robot_1
    #             ee_position = self._get_ee_position(robot)[env_id]
    #             dist = torch.norm(obj_pos.float() - ee_position.float(), p=2, dim=-1)
                
    #             reward = 2*torch.exp(-self.cfg.dist_reward_scale * dist)

               
    #             final_rewards['robot_2'][env_id] = self._get_r2_stationary_rew(env_id)
                
    #             final_rewards[agent_name][env_id] = reward

    #         if phase == Phases.GRIP_1_OPEN:
    #             # concetrate on opening the gripper
    #             agent_name = 'robot_1'
    #             gripper_width = self.phase_detector.get_gripper_width(self.robot_1)[env_id]
    #             final_rewards[agent_name][env_id] = self.cfg.dist_reward_scale * gripper_width
              
    #             final_rewards['robot_2'][env_id] = self._get_r2_stationary_rew(env_id) 


    #         if phase == Phases.GRIP_1_CLOSE:
    #             agent_name = 'robot_1'
    #             gripper_width = self.phase_detector.get_gripper_width(self.robot_1)[env_id]
    #             final_rewards[agent_name][env_id] =  2*torch.exp(-self.cfg.dist_reward_scale * gripper_width)

              
    #             final_rewards['robot_2'][env_id] = self._get_r2_stationary_rew(env_id) 
    #         if phase == Phases.LIFT:
    #             agent_name = 'robot_1'
    #             obj_pos_env = obj_pos_all[env_id]
    #             height = obj_pos_env[2] - self.cfg.ground_height
    #             final_rewards[agent_name][env_id] = self.cfg.dist_reward_scale * height
                
    #             final_rewards['robot_2'][env_id] = self._get_r2_stationary_rew(env_id)
                
            
    #         if phase == Phases.REACH_GOAL_1:
    #             agent_name = "robot_1"
    #             robot = self.robot_1
    #             ee_position = self._get_ee_position(robot)[env_id]
    #             goal_pos = self.goal_pos[env_id]
    #             dist = torch.norm(goal_pos.float() - ee_position.float(), dim=-1)
    #             reward = torch.exp(-self.cfg.dist_reward_scale * dist)
    #             final_rewards[agent_name][env_id] = reward
    #             final_rewards['robot_2'][env_id] = self._get_r2_stationary_rew(env_id)
                

    #         if phase == Phases.REACH_GOAL_2:
    #             agent_name = "robot_2"
    #             robot = self.robot_2
    #             ee_position = self._get_ee_position(robot)[env_id]
    #             goal_pos = self.goal_pos[env_id]
    #             dist = torch.norm(goal_pos.float() - ee_position.float(), dim=-1)
    #             reward = torch.exp(-self.cfg.dist_reward_scale * dist)
    #             final_rewards[agent_name][env_id] = reward

    #         if phase == Phases.GRIP_2:
    #             agent_name = 'robot_2'
    #             is_holding_obj = self.phase_detector.is_holding_object(
    #                 obj_pos_all, self.robot_2)[env_id]
    #             if is_holding_obj:
    #                 final_rewards[agent_name][env_id] = 10
            
    #         if phase == Phases.RELEASE_1:
    #             agent_name = 'robot_2'
    #             is_holding_obj = self.phase_detector.is_holding_object(
    #                 obj_pos_all, self.robot_2)[env_id]
    #             if is_holding_obj:
    #                 final_rewards[agent_name][env_id] = 10

    #             agent_name = 'robot_1'
    #             is_holding_obj = self.phase_detector.is_holding_object(
    #                 obj_pos_all, self.robot_1)[env_id]
    #             if not is_holding_obj:
    #                 final_rewards[agent_name][env_id] = 10


    #         if phase == Phases.END:
    #             agent_name = "robot_1"
    #             robot = self.robot_1
    #             ee_position = self._get_ee_position(robot)[env_id]
    #             goal_pos = self.r1_init_pos
    #             dist = torch.norm(goal_pos.float() - ee_position.float(), dim=-1)
    #             reward = torch.exp(-self.cfg.dist_reward_scale * dist)
    #             final_rewards[agent_name][env_id] = reward

    #     return final_rewards


    def _get_rewards(self):
        phases_one_hot = self._get_phase()  # shape (num_envs, num_phases)

        num_envs, num_phases = phases_one_hot.shape

        device = self.robot_1.data.device
        obj_pos = self._get_obj_pos()  # (num_envs, 3)
        ee_1 = self._get_ee_position(self.robot_1)
        ee_2 = self._get_ee_position(self.robot_2)
        goal_pos = self.goal_pos

        rewards = torch.zeros((num_envs, num_phases), device=device)

        # phase 0: REACH_OBJ
        dist = torch.norm(obj_pos - ee_1, dim=-1)
        rewards[:, Phases.REACH_OBJ.value] = 2 * torch.exp(-self.cfg.dist_reward_scale * dist)

        # phase 1: GRIP_1_OPEN
        gripper_width = self.phase_detector.get_gripper_width(self.robot_1)
        rewards[:, Phases.GRIP_1_OPEN.value] = self.cfg.dist_reward_scale * gripper_width

        # phase 2: GRIP_1_CLOSE
        rewards[:, Phases.GRIP_1_CLOSE.value] = 4 * torch.exp(-self.cfg.dist_reward_scale * gripper_width)

        # phase 3: LIFT
        height = obj_pos[:, 2] - self.cfg.ground_height
        rewards[:, Phases.LIFT.value] = self.cfg.dist_reward_scale * height

        # phase 4: REACH_GOAL_1
        dist_goal1 = torch.norm(goal_pos - ee_1, dim=-1)
        rewards[:, Phases.REACH_GOAL_1.value] = torch.exp(-self.cfg.dist_reward_scale * dist_goal1)

        # phase 5: REACH_GOAL_2
        dist_goal2 = torch.norm(goal_pos - ee_2, dim=-1)
        rewards[:, Phases.REACH_GOAL_2.value] = torch.exp(-self.cfg.dist_reward_scale * dist_goal2)

        # phase 6: GRIP_2
        holding_2 = self.phase_detector.is_holding_object(obj_pos, self.robot_2)
        rewards[:, Phases.GRIP_2.value] = 10 * holding_2.float()

        # phase 7: RELEASE_1
        holding_1 = self.phase_detector.is_holding_object(obj_pos, self.robot_1)
        both_condition = (~holding_1) & holding_2
        rewards[:, Phases.RELEASE_1.value] = 10 * both_condition.float()

        # phase 8: END
        
        dist_home = torch.norm(ee_1 - self.r1_init_pos, dim=-1)
        rewards[:, Phases.END.value] = torch.exp(-self.cfg.dist_reward_scale * dist_home)

        # final reward for robot_1: dot product (envs × phases) ⊙ (envs × phases)
        final_rewards_r1 = torch.sum(phases_one_hot * rewards, dim=1)  # (num_envs,)
        final_rewards_r2 = torch.zeros((self.num_envs,), dtype=torch.float, device=self.robot_1.data.device)

        return {
            "robot_1": final_rewards_r1,
            "robot_2": final_rewards_r2
        }

    def _get_dones(self):
        self._compute_intermediate_values()
        obj_pos_z = self.object.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        fallen = obj_pos_z < self.cfg.fall_z_threshold
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return (
            {agent: fallen.clone() for agent in self.cfg.possible_agents},
            {agent: timeout.clone() for agent in self.cfg.possible_agents},
        )

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # Reset object pose with some noise
        pos_noise = self.cfg.reset_position_noise * sample_uniform(-1, 1, (len(env_ids), 3), self.device)
        rot_noise = self.cfg.reset_rot_noise * sample_uniform(-1, 1, (len(env_ids), 2), self.device)
        new_pos = self.scene.env_origins[env_ids] + pos_noise
        new_rot = randomize_rotation(rot_noise[:, 0], rot_noise[:, 1])
        new_rot[0] = torch.tensor([0.5, 0.5, 0.5, 0.5])
       
        
        
        self.object.write_root_pose_to_sim(torch.cat((new_pos, new_rot), dim=-1), env_ids)
        self._compute_intermediate_values()



@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1):
    x_unit = torch.tensor([1.0, 0.0, 0.0], device="cuda").repeat((rand0.shape[0], 1))
    y_unit = torch.tensor([0.0, 1.0, 0.0], device="cuda").repeat((rand0.shape[0], 1))
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit), quat_from_angle_axis(rand1 * np.pi, y_unit)
    )

