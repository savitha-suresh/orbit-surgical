import torch
import numpy as np
from isaaclab.envs import DirectMARLEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import sample_uniform, quat_from_angle_axis, quat_mul, saturate

from isaaclab.utils.math import subtract_frame_transforms

from .joint_pos_env_cfg import BlockHandoverEnvCfg
from .phase_detector import Phases, PhaseDetector
from isaaclab.markers import VisualizationMarkers




class DualArmHandoverEnv(DirectMARLEnv):
    cfg: BlockHandoverEnvCfg

    def __init__(self, cfg: BlockHandoverEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.phase_detector = PhaseDetector(cfg, self)
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
        self.current_phases[:, Phases.REACH_P1.value] = 1.0



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


        self.goal_markers = VisualizationMarkers(self.cfg.p1_pos_cfg)
        self.goal_markers_obj = VisualizationMarkers(self.cfg.obj_pos_cfg)
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
        obj_pos = self._get_obj_pos()
        self.goal_markers_obj.visualize(self._get_obj_pos())
        p1_pos = self.get_p1_pos(obj_pos)
        self.goal_markers.visualize(p1_pos)
        

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
        
        #print(self.phase_detector.get_gripper_width(self.robot_1))
        #print(self.current_phases)
        # Process each robot separately
        for robot_name in self.cfg.possible_agents:
            robot = self.scene.articulations[robot_name]
 
            # Resolve joint IDs
            joint_ids = [robot.joint_names.index(name) for name in robot.joint_names]
  
            joint_pos_rel = robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]
             
            joint_vel_rel = robot.data.joint_vel[:, joint_ids] - robot.data.default_joint_vel[:, joint_ids]
            
            obj_pos = self._get_obj_pos()
            ee_pose = self._get_ee_pose(robot)
            # Concatenate all observations for this robot
            obs_list = [
                joint_pos_rel,
                joint_vel_rel,
                obj_pos,
                ee_pose,
                self.goal_pos,
                self.goal_rot,
                self.get_p1_pos(obj_pos)
            ]
            
            # Concatenate along the feature dimension
            robot_obs = torch.cat(obs_list, dim=-1)
            
           
            observations[robot_name] = robot_obs
    
        return observations
    

    def _apply_action(self):

        
        
        
        
        self.robot_1_curr_targets[:, self.actuated_dof_indices] = scale(
            self.actions["robot_1"],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.robot_1_curr_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.robot_1_curr_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.robot_1_prev_targets[:, self.actuated_dof_indices]
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

        

        self.robot_1_prev_targets[:, self.actuated_dof_indices] = self.robot_1_curr_targets[
            :, self.actuated_dof_indices
        ]
        self.robot_2_prev_targets[:, self.actuated_dof_indices] = self.robot_2_curr_targets[
            :, self.actuated_dof_indices
        ]
        # self.robot_1.set_joint_position_target(
        #     self.robot_1_curr_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        # )

        # self.count+=1
       
        # if self.count > 200:
        #     print("increasing count")
        #     self.robot_1_curr_targets[:, 2] += 0.16
            
            
        # if self.count > 500:
        #     print("beinding")
        #     self.robot_1_curr_targets[:, 4] += -50
        
        self.robot_1.set_joint_position_target(
            # 0. - Swings the arm side to side (base yaw) - X position of ee
            # 1. - Moves the arm up/down (pitch motion) - Y axis moving ee
            # 2. Length
            # 3. Rolls instrument around tool shaft
            # 4. Bends the tip up/down
            # 5. Turns the tip side to side
            # 6. gripper 
            # 7. Gripper length
            self.robot_1_curr_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices,
        )
        
        # self.robot_2.set_joint_position_target(
        #     self.robot_2_curr_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        # )
        
    
    
    def _get_states(self):
        
        # same as observation but both agents in one vector
        obs = self._get_observations()
        
        return torch.cat([obs[agent] for agent in self.cfg.possible_agents], dim=-1)
    

    def _get_phase(self):
        
        self.current_phases, phase_indices, phase_regressed_mask, phase_same_mask = self.phase_detector.get_phases(
            agents=[self.robot_1, self.robot_2],
            obj_position=self._get_obj_pos(),
            batch_size=self.num_envs,
            goal_position=self.goal_pos,
            prev_phases = self.current_phases.clone()
        )
        
        return self.current_phases, phase_indices, phase_regressed_mask, phase_same_mask

    def _get_obj_pos(self):
        #return self.object.data.root_pos_w - self.scene.env_origin
        pos_all = self.object.data.root_pos_w
        pos_new = pos_all.clone()
        pos_new[:, 2] += 0.01
        return  pos_new
    
    def get_p1_pos(self, obj_position, approach_angle=35):
        """
        Create P1 at 45-degree approach angle
        """
        angle_rad = torch.deg2rad(torch.tensor(approach_angle))
        
        # Distance from object (adjust this based on your needs)
        approach_distance = 0.06  # 5cm approach distance
        
        # Calculate P1 position at 45-degree angle
        p1_pos = obj_position.clone()
        p1_pos[:, 0] += approach_distance * torch.cos(angle_rad)  # X offset
        p1_pos[:, 2] += approach_distance * torch.sin(angle_rad)  # Z offset (height)
        
        return p1_pos
    
    def _get_r2_stationary_rew(self, env_id):
        ee_position_2 = self._get_ee_position(self.robot_2)[env_id]
        goal_pos = self.r2_init_pos[env_id]
        dist_2 = torch.norm(goal_pos.float() - ee_position_2.float(), p=2, dim=-1)
        
        reward_2 = 2*torch.exp(-self.cfg.dist_reward_scale * dist_2)
        
        return reward_2
    

    def is_point_between_parallel_lines(self, obj_position, p1_pos, ee_pos, margin=0.05):
        
        
        x_min = torch.min(obj_position[:, 0], p1_pos[:, 0]) - margin
        x_max = torch.max(obj_position[:, 0], p1_pos[:, 0]) + margin
        return (ee_pos[:, 0] >= x_min) & (ee_pos[:, 0] <= x_max)



    def _get_rewards(self):
        phases_one_hot, phase_indices, phase_regressed_mask, phase_same_mask = self._get_phase()  # shape (num_envs, num_phases)
        num_envs, num_phases = phases_one_hot.shape
        device = self.robot_1.data.device
        obj_pos = self._get_obj_pos()  # (num_envs, 3)
        ee_1 = self._get_ee_position(self.robot_1)
        ee_2 = self._get_ee_position(self.robot_2)
        p1_pos = self.get_p1_pos(obj_pos)
        goal_pos = self.goal_pos

        rewards = torch.zeros((num_envs, num_phases), device=device)


        dist_p1_ee = torch.norm(p1_pos - ee_1, dim=-1)
        
        rewards[:, Phases.REACH_P1.value] = 2 * torch.exp(-50 * dist_p1_ee) 
        env_ids = torch.arange(self.num_envs)
        mask = (dist_p1_ee <= self.phase_detector.CLOSE_THRESHOLD) & self.not_visited_mask[env_ids, Phases.REACH_P1.value]

        #print("visited", self.not_visited_mask, mask, self.not_visited_mask[env_ids, Phases.REACH_P1.value])
        self.not_visited_mask[mask, Phases.REACH_P1.value] = False
        rewards[mask, Phases.REACH_OBJ.value] += 2
        #print("rew", rewards)
        # phase 0: REACH_OBJ
        z_dist = torch.norm(obj_pos - ee_1, dim=-1)
        
        rewards[:, Phases.REACH_OBJ.value] +=  2+(
            2* torch.exp(-40 * z_dist)
        )

        # print("dist obj ee", dist)
        #rewards[:, Phases.REACH_OBJ.value] = 2 * torch.exp(-self.cfg.dist_reward_scale_reach_obj * z_dist) * self.cfg.reward_scale
    
        # phase 1: GRIP_1_OPEN
        gripper_width = self.phase_detector.get_gripper_width(self.robot_1)
        rewards[:, Phases.GRIP_1_OPEN.value] = self.cfg.dist_reward_scale * gripper_width * self.cfg.reward_scale
    
        # phase 2: GRIP_1_CLOSE
        rewards[:, Phases.GRIP_1_CLOSE.value] = 40 * torch.exp(-self.cfg.dist_reward_scale * gripper_width) * self.cfg.reward_scale
        # phase 3: LIFT
        height = obj_pos[:, 2] - self.cfg.ground_height
        rewards[:, Phases.LIFT.value] = self.cfg.dist_reward_scale * height * self.cfg.reward_scale
        

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
        

        #print(rewards)
        final_rewards_r1 = torch.sum(phases_one_hot * rewards  , dim=1)  # (num_envs,)
        final_rewards_r1[phase_regressed_mask] += self.cfg.phase_regressed_penalty
        
        final_rewards_r2 = torch.zeros((self.num_envs,), dtype=torch.float, device=self.robot_1.data.device)
        self.phase_visit_counts[torch.arange(self.num_envs), phase_indices] += 1
        
        
        #print("rewards", final_rewards_r1)
        
        return {
            "robot_1": final_rewards_r1,
            "robot_2": final_rewards_r2
        }

    def _get_dones(self):
        self._compute_intermediate_values()
        obj_pos_z = self.object.data.root_pos_w[:, 2]
        fallen = obj_pos_z < self.cfg.fall_z_threshold
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return (
            {agent: fallen.clone() for agent in self.cfg.possible_agents},
            {agent: timeout.clone() for agent in self.cfg.possible_agents},
        )



    def _reset_idx(self, env_ids):
        
        super()._reset_idx(env_ids)
       
        
        # Set to True — all phases are not visited initially
        self.not_visited_mask = torch.ones((self.num_envs, len(Phases)), dtype=torch.bool, device=self.device)
        # In __init__ or reset()
        self.phase_visit_counts = torch.zeros((self.num_envs, len(Phases)), device=self.device, dtype=torch.float)

        # Reset object pose with some noise
        pos_noise = self.cfg.reset_position_noise * sample_uniform(-1, 1, (len(env_ids), 3), self.device)
        rot_noise = self.cfg.reset_rot_noise * sample_uniform(-1, 1, (len(env_ids), 2), self.device)
        new_pos = self.scene.env_origins[env_ids] + pos_noise
        new_rot = randomize_rotation(rot_noise[:, 0], rot_noise[:, 1])
        # new_rot[0] = torch.tensor([0.5, 0.5, 0.5, 0.5])
        self.current_phases[:, Phases.REACH_P1.value] = 1.0

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
        
        
        self.object.write_root_pose_to_sim(torch.cat((new_pos, new_rot), dim=-1), env_ids)
        
        self.count = 0
        
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

