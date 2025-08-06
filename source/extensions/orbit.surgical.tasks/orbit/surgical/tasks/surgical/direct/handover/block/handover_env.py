import torch
import numpy as np
from isaaclab.envs import DirectMARLEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import sample_uniform, quat_from_angle_axis, quat_mul, saturate

from isaaclab.utils.math import subtract_frame_transforms, quat_rotate

from .joint_pos_env_cfg import BlockHandoverEnvCfg
from .phase_detector import Phases, PhaseDetector, log_if
from isaaclab.markers import VisualizationMarkers


def quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert normalized quaternion (w, x, y, z) to rotation matrix."""
    qw, qx, qy, qz = quat.unbind(-1)

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    m00 = 1 - 2 * (yy + zz)
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)

    m10 = 2 * (xy + wz)
    m11 = 1 - 2 * (xx + zz)
    m12 = 2 * (yz - wx)

    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = 1 - 2 * (xx + yy)

    return torch.stack([
        torch.stack([m00, m01, m02], dim=-1),
        torch.stack([m10, m11, m12], dim=-1),
        torch.stack([m20, m21, m22], dim=-1),
    ], dim=-2)  # (..., 3, 3)


class DualArmHandoverEnv(DirectMARLEnv):
    cfg: BlockHandoverEnvCfg

    def __init__(self, cfg: BlockHandoverEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.phase_detector = PhaseDetector(cfg, self)
        self.ee_link_name = self.cfg.ee_link_name
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        

        self.r1_init_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.r1_init_pos[:, :] = torch.tensor([0.18, 0.0, 0.15], device=self.device)
        self.r2_init_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.r2_init_pos[:, :] = torch.tensor([-0.18, 0.0, 0.15], device=self.device)
        self.current_phases = torch.zeros((self.num_envs, len(Phases)), dtype=torch.float, device=self.device)
        self.current_phases[:, Phases.REACH_P1.value] = 1.0


        self.phase_regressed_mask = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
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
        self.markers_goal = VisualizationMarkers(self.cfg.goal_pos_cfg)
        self.ee_tgt_marker = VisualizationMarkers(self.cfg.ee_tgt_pos_cfg)
        self.grip_tgt_marker = VisualizationMarkers(self.cfg.grip_tgt_pos_cfg)
        self.grip_lnk_marker = VisualizationMarkers(self.cfg.grip_lnk_pos_cfg)
        self.tip_1_marker = VisualizationMarkers(self.cfg.tip_1_cfg)
        self.tip_2_marker = VisualizationMarkers(self.cfg.tip_2_cfg)
        self.grp_pt_1_marker = VisualizationMarkers(self.cfg.grp_pt_1_cfg)
        self.grp_pt_2_marker = VisualizationMarkers(self.cfg.grp_pt_2_cfg)
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
        self.goal_markers_obj.visualize(obj_pos)
        p1_pos = self.get_p1_pos(obj_pos)
        self.goal_markers.visualize(p1_pos)
        goal_pos = self.get_goal_pos(obj_pos)
        self.markers_goal.visualize(goal_pos)
        self.ee_tgt_marker.visualize(self.get_obj_grip_pos())
        self.grip_tgt_marker.visualize(self.get_obj_griplnk_tgt_pos())
        self.grip_lnk_marker.visualize(self.get_gripper_link_pos(self.robot_1))
        grip_pos = self.get_gripper_tip_positions(self.robot_1)
        
        self.tip_1_marker.visualize(grip_pos[0])
        self.tip_2_marker.visualize(grip_pos[1])
        grip_end_pts = self.get_gripper_target_points()
        self.grp_pt_1_marker.visualize(grip_end_pts[0])
        self.grp_pt_2_marker.visualize(grip_end_pts[1])

        

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
    
    def get_gripper_link_pos(self, robot):
        gripper_link_pos = robot.data.body_pos_w[:, robot.find_bodies(self.cfg.gripper_name)[0]]
        return gripper_link_pos.squeeze(1)

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
            
            # Keep relative joint positions (this is already good)
            joint_pos_rel = robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]
            joint_vel_rel = robot.data.joint_vel[:, joint_ids] - robot.data.default_joint_vel[:, joint_ids]
            
            # Get current poses
            obj_pos = self._get_obj_pos()
            ee_pose = self._get_ee_position(robot)
            gripper_link_pos = self.get_gripper_link_pos(robot)
            
            # RELATIVE OBSERVATIONS - Key changes here
            # 1. End-effector to object vector (relative position)
            ee_to_obj = obj_pos - ee_pose  # Assuming ee_pose has position in first 3 dims
            
            # 2. End-effector to goal vector (relative position)
            goal_pos = self.get_goal_pos(obj_pos)
            ee_to_goal = goal_pos - ee_pose
            
            # 3. Object to goal vector (relative position)
            obj_to_goal = goal_pos - obj_pos
            
            # 4. Distance metrics (scale-invariant)
            ee_to_obj_distance = torch.norm(ee_to_obj, dim=-1, keepdim=True)
            ee_to_goal_distance = torch.norm(ee_to_goal, dim=-1, keepdim=True)
            obj_to_goal_distance = torch.norm(obj_to_goal, dim=-1, keepdim=True)
            
            # 5. Normalized direction vectors
            ee_to_obj_dir = ee_to_obj / (ee_to_obj_distance + 1e-8)
            ee_to_goal_dir = ee_to_goal / (ee_to_goal_distance + 1e-8)
            obj_to_goal_dir = obj_to_goal / (obj_to_goal_distance + 1e-8)
            
            
            
            # 7. Get relative position to waypoints
            p1_pos = self.get_p1_pos(obj_pos)
            ee_to_p1 = p1_pos - ee_pose[:, :3]
            ee_to_p1_dir = ee_to_p1 / (torch.norm(ee_to_p1, dim=-1, keepdim=True) + 1e-8)


            grip_pos = self.get_gripper_tip_positions(self.robot_1)
            grp_tgt_pos = self.get_gripper_target_points()
            grp1_tgt_pos = grp_tgt_pos[0]
            grp2_tgt_pos = grp_tgt_pos[1]
            grip1_pos = grip_pos[0]
            grip2_pos = grip_pos[1]

            ee1_to_grip = grp1_tgt_pos - grip1_pos
            ee1_to_grip_dir = ee1_to_grip / (torch.norm(ee1_to_grip, dim=-1, keepdim=True) + 1e-8)
            ee1_to_grip_distance = torch.norm(ee1_to_grip, dim=-1, keepdim=True)

            
            ee2_to_grip = grp2_tgt_pos - grip2_pos
            ee2_to_grip_dir = ee2_to_grip / (torch.norm(ee2_to_grip, dim=-1, keepdim=True) + 1e-8)
            ee2_to_grip_distance = torch.norm(ee2_to_grip, dim=-1, keepdim=True)


            gripper_target_pos = self.get_gripper_link_target_pos()
            gripper_to_target = gripper_target_pos - gripper_link_pos
            gripper_to_target_dir = gripper_to_target / (torch.norm(gripper_to_target, dim=-1, keepdim=True) + 1e-8)
            gripper_to_target_distance = torch.norm(gripper_to_target, dim=-1, keepdim=True)


            obj_griplink_target_pos = self.get_obj_griplnk_tgt_pos()
            objgripper_to_target = obj_griplink_target_pos - gripper_link_pos
            objgripper_to_target_dir = objgripper_to_target / (torch.norm(objgripper_to_target, dim=-1, keepdim=True) + 1e-8)
            objgripper_to_target_distance = torch.norm(objgripper_to_target, dim=-1, keepdim=True)


            # Concatenate RELATIVE observations
            obs_list = [
                joint_pos_rel,                    # Joint positions (already relative)
                joint_vel_rel,                    # Joint velocities (already relative)
                ee_to_obj,                        # Vector from EE to object
                ee_to_goal,                       # Vector from EE to goal
                obj_to_goal,                      # Vector from object to goal
                ee_to_p1,                         # Vector from EE to waypoint
                ee1_to_grip, 
                ee2_to_grip,
                gripper_to_target,
                objgripper_to_target,
                ee_to_obj_distance,               # Distance to object
                ee_to_goal_distance,              # Distance to goal
                ee1_to_grip_distance, 
                ee2_to_grip_distance,
                gripper_to_target_distance,
                objgripper_to_target_distance,
                ee_to_obj_dir,                    # Direction to object (normalized)
                ee_to_goal_dir,                   # Direction to goal (normalized)
                obj_to_goal_dir,                  # Direction object should move
                gripper_to_target_dir,
                objgripper_to_target_dir,
                ee_to_p1_dir,                     # Direction to waypoint
                ee1_to_grip_dir,
                ee2_to_grip_dir,
                self.not_visited_mask,            # Task phase info
                self.phase_regressed_mask.unsqueeze(1)
            ]
            
            
            # Concatenate along the feature dimension
            robot_obs = torch.cat(obs_list, dim=-1)
            observations[robot_name] = robot_obs
    
        return observations
    

    # def _apply_action(self):

        
        
        
        
    #     self.robot_1_curr_targets[:, self.actuated_dof_indices] = scale(
    #         self.actions["robot_1"],
    #         self.hand_dof_lower_limits[:, self.actuated_dof_indices],
    #         self.hand_dof_upper_limits[:, self.actuated_dof_indices],
    #     )
    #     self.robot_1_curr_targets[:, self.actuated_dof_indices] = (
    #         self.cfg.act_moving_average * self.robot_1_curr_targets[:, self.actuated_dof_indices]
    #         + (1.0 - self.cfg.act_moving_average) * self.robot_1_prev_targets[:, self.actuated_dof_indices]
    #     )

        
    #     self.robot_1_curr_targets[:, self.actuated_dof_indices] = saturate(
    #         self.robot_1_curr_targets[:, self.actuated_dof_indices],
    #         self.hand_dof_lower_limits[:, self.actuated_dof_indices],
    #         self.hand_dof_upper_limits[:, self.actuated_dof_indices],
    #     )

        
    #     self.robot_2_curr_targets[:, self.actuated_dof_indices] = scale(
    #         self.actions["robot_2"],
    #         self.hand_dof_lower_limits[:, self.actuated_dof_indices],
    #         self.hand_dof_upper_limits[:, self.actuated_dof_indices],
    #     )
    #     self.robot_2_curr_targets[:, self.actuated_dof_indices] = (
    #         self.cfg.act_moving_average * self.robot_2_curr_targets[:, self.actuated_dof_indices]
    #         + (1.0 - self.cfg.act_moving_average) * self.robot_2_prev_targets[:, self.actuated_dof_indices]
    #     )
    #     self.robot_2_curr_targets[:, self.actuated_dof_indices] = saturate(
    #         self.robot_2_curr_targets[:, self.actuated_dof_indices],
    #         self.hand_dof_lower_limits[:, self.actuated_dof_indices],
    #         self.hand_dof_upper_limits[:, self.actuated_dof_indices],
    #     )

        

    #     self.robot_1_prev_targets[:, self.actuated_dof_indices] = self.robot_1_curr_targets[
    #         :, self.actuated_dof_indices
    #     ]
    #     self.robot_2_prev_targets[:, self.actuated_dof_indices] = self.robot_2_curr_targets[
    #         :, self.actuated_dof_indices
    #     ]
    #     # self.robot_1.set_joint_position_target(
    #     #     self.robot_1_curr_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
    #     # )

    #     # self.count+=1
       
    #     # if self.count > 200:
    #     #     print("increasing count")
    #     #     self.robot_1_curr_targets[:, 2] += 0.16
            
            
    #     # if self.count > 500:
    #     #     print("beinding")
    #     #     self.robot_1_curr_targets[:, 4] += -50
        
    #     self.robot_1.set_joint_position_target(
    #         # 0. - Swings the arm side to side (base yaw) - X position of ee
    #         # 1. - Moves the arm up/down (pitch motion) - Y axis moving ee
    #         # 2. Length
    #         # 3. Rolls instrument around tool shaft
    #         # 4. Bends the tip up/down
    #         # 5. Turns the tip side to side
    #         # 6. gripper 
    #         # 7. Gripper length
    #         self.robot_1_curr_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices,
    #     )
        
    #     # self.robot_2.set_joint_position_target(
    #     #     self.robot_2_curr_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
    #     # )
        
    def _apply_action(self):
        """
        Modified to use relative/delta actions instead of absolute positions
        """
        
        # RELATIVE ACTIONS - Key changes here
        # Actions now represent deltas/changes rather than absolute targets
        
        # Scale actions to reasonable delta ranges (e.g., -0.1 to 0.1 radians per step)
        base_scale = 0.19
        reduced_scale = 0.12
        action_scale = torch.full((self.num_envs, 1), base_scale, device=self.device)

        # Check if GRIP_OPEN phase is active (1 or True)
        grip_open_active = ~self.not_visited_mask[:, Phases.GRIP_1_OPEN.value] & self.not_visited_mask[:, Phases.GRIP_1_CLOSE.value]
        lift_active = self.current_phases[:, Phases.LIFT.value].bool()
        # Apply reduced scale where grip is open
        action_scale[grip_open_active] = reduced_scale

        
        # Robot 1 - Apply relative changes
        # Scale actions from [-1, 1] to [-action_scale, action_scale]
        action_deltas_1 = self.actions["robot_1"] * action_scale
        
        # Update targets by adding deltas to CURRENT positions (not previous targets)
        current_joint_pos = self.robot_1.data.joint_pos[:, self.actuated_dof_indices]
        self.robot_1_curr_targets[:, self.actuated_dof_indices] = (
            current_joint_pos + action_deltas_1
        )
        
        # Apply moving average for smoothing
        self.robot_1_curr_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.robot_1_prev_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.robot_1_curr_targets[:, self.actuated_dof_indices]
        )
        
        # Clamp to joint limits
        self.robot_1_curr_targets[:, self.actuated_dof_indices] = saturate(
            self.robot_1_curr_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        
        # Robot 2 - Same approach
        action_deltas_2 = self.actions["robot_2"] * action_scale
        
        current_joint_pos_2 = self.robot_2.data.joint_pos[:, self.actuated_dof_indices]
        self.robot_2_curr_targets[:, self.actuated_dof_indices] = (
            current_joint_pos_2 + action_deltas_2
        )
        
        self.robot_2_curr_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.robot_2_prev_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.robot_2_curr_targets[:, self.actuated_dof_indices]
        )
        
        self.robot_2_curr_targets[:, self.actuated_dof_indices] = saturate(
            self.robot_2_curr_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        

        gripper_dof_idxs = self.actuated_dof_indices[-2:]
        grip_close_mask = self.current_phases[:, Phases.GRIP_1_CLOSE.value].bool() | ~self.not_visited_mask[:, Phases.REACH_OBJ_GRIP.value]
        grip_envs = torch.nonzero(grip_close_mask).squeeze(-1)
        non_grip_envs = torch.nonzero(~grip_close_mask).squeeze(-1)

        # Copy over previous targets for gripper-close envs
        if grip_envs.numel() > 0:
            closed_position = 0.0  # or whatever your closed position should be
            #current_targets = self.robot_1.data.joint_pos_target.clone()
            self.robot_1_curr_targets[grip_envs[:, None], -1] = 0.07
            self.robot_1_curr_targets[grip_envs[:, None], -2] = -0.07
            #current_targets[grip_envs[:, None], gripper_dof_idxs] = closed_position
            
            
            # Set high position gains for immediate response
            self.robot_1.set_joint_position_target(
                self.robot_1_curr_targets[grip_envs[:, None], self.actuated_dof_indices],
                env_ids=grip_envs, joint_ids=self.actuated_dof_indices)
            
            # Option 2: Set velocity directly for controlled closure
            # current_velocities = self.robot_1.data.joint_vel_target.clone()
            # close_velocity = -20.0  # negative for closing, adjust magnitude as needed
            # current_velocities[grip_envs[:, None], self.actuated_dof_indices[:-2]] = close_velocity
            # self.robot_1.set_joint_velocity_target(
            #     current_velocities[grip_envs[:, None], self.actuated_dof_indices[:-2]],
            #     env_ids=grip_envs, joint_ids=self.actuated_dof_indices[:-2])
            #self.robot_1_curr_targets[grip_envs[:, None], :] = 0.0

            # # Set only gripper joints for gripper-close envs
            # self.robot_1.set_joint_position_target(
            #     self.robot_1_curr_targets[grip_envs[:, None], gripper_dof_idxs],
            #     env_ids=grip_envs,
            #     joint_ids=gripper_dof_idxs,
            # )

        # For the remaining envs, set all actuated joints
        if non_grip_envs.numel() > 0:
            self.robot_1.set_joint_position_target(
                self.robot_1_curr_targets[non_grip_envs[:, None], self.actuated_dof_indices],
                env_ids=non_grip_envs,
                joint_ids=self.actuated_dof_indices,
            )

        # Store previous targets for next iteration
        self.robot_1_prev_targets[:, self.actuated_dof_indices] = self.robot_1_curr_targets[:, self.actuated_dof_indices]
        self.robot_2_prev_targets[:, self.actuated_dof_indices] = self.robot_2_curr_targets[:, self.actuated_dof_indices]
        
        # Apply the targets
        # self.robot_1.set_joint_position_target(
        #     self.robot_1_curr_targets[:, self.actuated_dof_indices], 
        #     joint_ids=self.actuated_dof_indices,
        # )
    
    
        # Uncomment when ready to control robot_2
        # self.robot_2.set_joint_position_target(
        #     self.robot_2_curr_targets[:, self.actuated_dof_indices], 
        #     joint_ids=self.actuated_dof_indices
        # )


    

    def _get_states(self):
        
        # same as observation but both agents in one vector
        obs = self._get_observations()
        
        return torch.cat([obs[agent] for agent in self.cfg.possible_agents], dim=-1)
    

    def _get_phase(self):
        
        obj_pos = self._get_obj_pos()
        self.current_phases, phase_indices, phase_regressed_mask, phase_same_mask = self.phase_detector.get_phases(
            agents=[self.robot_1, self.robot_2],
            obj_position=self._get_obj_pos(),
            batch_size=self.num_envs,
            goal_position=self.get_goal_pos(obj_pos),
            prev_phases = self.current_phases.clone()
        )
        self.phase_regressed_mask = phase_regressed_mask
        return self.current_phases, phase_indices, phase_regressed_mask, phase_same_mask

    def get_abs_obj_pos(self):
        return self.object.data.root_pos_w
    
    def _get_obj_pos(self):
        base_pos = self.object.data.root_pos_w         # (N, 3)
        base_rot = self.object.data.root_quat_w         # (N, 4)
        
        local_offset = torch.tensor([[-0.00012, 0.005, 0.02]], device=base_pos.device)  # (1, 3)
        N = base_pos.shape[0]
        local_offset = local_offset.expand(N, -1) 
        rot_mat = quat_to_matrix(base_rot)             # (N, 3, 3)
        
        offset_world = torch.bmm(rot_mat, local_offset.unsqueeze(-1)).squeeze(-1)  # (N, 3)
        return base_pos + offset_world
    

    def get_dist_toadjust_griplink(self):
        gripper_link_pos = self.get_gripper_link_pos(self.robot_1)
        ee_1 = self._get_ee_position(self.robot_1)
        return torch.norm(gripper_link_pos - ee_1, dim=-1)
    
    
    def get_obj_griplnk_tgt_pos(self):
        # Get base grip point (already rotation-aware)
        grip_pos = self._get_obj_pos()

        # Get object orientation
        base_rot = self.object.data.root_quat_w         # (N, 4)
        rot_mat = quat_to_matrix(base_rot)             # (N, 3, 3)
        z_axis = rot_mat[:, :, 2]                      # Object's local Z in world

        # Get vertical offset based on gripper
        dist_offset = self.get_dist_toadjust_griplink()  # (N,)
        grip_pos_adjusted = grip_pos + z_axis * dist_offset.unsqueeze(-1)

        return grip_pos_adjusted
    
    def get_obj_grip_pos(self):
        base_pos = self.object.data.root_pos_w          # (N, 3)
        base_rot = self.object.data.root_quat_w          # (N, 4)
        # x,y are for the needle
        local_offset = torch.tensor([[-0.00012, 0.005, 0.003]], device=base_pos.device)  # (1, 3)
        N = base_pos.shape[0]
        local_offset = local_offset.expand(N, -1)

        # Convert quaternion to rotation matrix
        rot_mat = quat_to_matrix(base_rot)              # (N, 3, 3)

        # Apply offset in object local frame
        offset_world = torch.bmm(rot_mat, local_offset.unsqueeze(-1)).squeeze(-1)  # (N, 3)

        # Add to base pos
        return base_pos + offset_world

    def get_obj_rotation(self):
        # Assuming self.peg is your RigidObject instance
        # The quaternion is stored in the root_quat_w (world frame)
        obj_quat = self.object.data.root_quat_w  # Shape: (num_envs, 4) - [x, y, z, w]
        return obj_quat
    
    def get_gripper_tip_positions(self, robot, jaw_radius=0.01):
        # 1. Get gripper displacements
        gripper_pos = self.phase_detector.get_gripper_pos(robot)  # shape: (num_envs, 2)
        jaw_disp = gripper_pos * jaw_radius
        # 2. Get pose of tool_tip_link
        ee_pose = self._get_ee_pose(robot)
        tip_pos = ee_pose[:, :3]            # (num_envs, 4) as quaternion
        tip_rot = ee_pose[:, 3:]
        # 3. Convert rotation to matrix
        tip_rot_mat = quat_to_matrix(tip_rot)   # shape: (num_envs, 3, 3)

        # 4. Gripper opening is along local Y axis
        x_axis = tip_rot_mat[:, :, 0]                       # local X axis
        # Only use joint displacement, no jaw_length added
        gr1_tip_pos = tip_pos + x_axis * jaw_disp[:, 0:1]
        gr2_tip_pos = tip_pos + x_axis * jaw_disp[:, 1:2]
        return gr1_tip_pos, gr2_tip_pos
        

    def get_grp_tgt_distance(self, robot):
        grp_tip_pts = self.get_gripper_tip_positions(robot)
        grp_tgt_pts = self.get_gripper_target_points()
        return torch.norm(grp_tip_pts[0] - grp_tgt_pts[0], dim=-1), torch.norm(grp_tip_pts[1] - grp_tgt_pts[1], dim=-1)

    def get_gripper_target_points(self):
        displacement = 0.5
        jaw_radius = 0.01
        world_disp = displacement * jaw_radius
        
        grip_pt = self.get_obj_grip_pos()
        obj_quat = self.get_obj_rotation()
        peg_rot_mat = quat_to_matrix(obj_quat)         # (N, 3, 3)

    #   
        direction = peg_rot_mat[:, :, 0]  # (N, 3)

        # 3. Offset gripper points
        grip1 = grip_pt + world_disp * direction
        grip2 = grip_pt - world_disp * direction
        
        return grip1, grip2

        # world_disp = displacement * jaw_radius
        # grip1 = grip_pt - world_disp * direction
        # grip2 = grip_pt + world_disp * direction
        # return grip1, grip2
    
    def get_gripper_link_target_pos(self):
        obj_grip_pos = self.get_obj_grip_pos()
        pos_new = obj_grip_pos.clone()
        pos_new[:, 2] += self.get_dist_toadjust_griplink() # calculated by printing the distance
        return pos_new

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
    

    def get_goal_pos(self, obj_position, approach_angle=-215): 
        """
        Create P1 at 45-degree approach angle
        """
        # angle_rad = torch.deg2rad(torch.tensor(approach_angle))
        
        # # Distance from object (adjust this based on your needs)
        # approach_distance = 0.03  # 5cm approach distance
        
        num_envs = obj_position.shape[0]
        goal_position = torch.tensor([-0.02, 0.02, 0.04], device=obj_position.device).unsqueeze(0)  # shape (1, 3)
        goal_position = goal_position + self.scene.env_origins  # shape (num_envs, 3)
        return goal_position 
        
        
    
    
    def _get_r2_stationary_rew(self, env_id):
        ee_position_2 = self._get_ee_position(self.robot_2)[env_id]
        goal_pos = self.r2_init_pos[env_id]
        dist_2 = torch.norm(goal_pos.float() - ee_position_2.float(), p=2, dim=-1)
        
        reward_2 = 2*torch.exp(-self.cfg.dist_reward_scale * dist_2)
        
        return reward_2
    

    def is_point_between_parallel_lines(self, obj_position, p1_pos, ee_pos, margin=0.03):
        
        
        x_min = torch.min(obj_position[:, 0], p1_pos[:, 0]) - margin
        x_max = torch.max(obj_position[:, 0], p1_pos[:, 0]) + margin
        return (ee_pos[:, 0] >= x_min) & (ee_pos[:, 0] <= x_max)


    def is_point_between_parallel_lines_done(self, obj_position, p1_pos, ee_pos, margin=0.01):
        
        
        x_min = torch.min(obj_position[:, 0], p1_pos[:, 0]) + margin
        x_max = torch.max(obj_position[:, 0], p1_pos[:, 0]) - margin
        return (ee_pos[:, 0] >= x_min) & (ee_pos[:, 0] <= x_max)


    def _get_rewards(self):
        phases_one_hot, phase_indices, phase_regressed_mask, phase_same_mask = self._get_phase()  # shape (num_envs, num_phases)
        num_envs, num_phases = phases_one_hot.shape
        device = self.robot_1.data.device
        obj_pos = self._get_obj_pos()  # (num_envs, 3)
        obj_abs_pos = self.get_abs_obj_pos()
        ee_1 = self._get_ee_position(self.robot_1)
        ee_2 = self._get_ee_position(self.robot_2)
        p1_pos = self.get_p1_pos(obj_pos)
        goal_pos = self.get_goal_pos(obj_pos)
        gripper_link_pos = self.get_gripper_link_pos(self.robot_1)
        gripper_link_tgt = self.get_gripper_link_target_pos()
        obj_griplink_pos = self.get_obj_griplnk_tgt_pos()
        obj_grip_pos = self.get_obj_grip_pos()
        log_if(not self.cfg.is_training, "phase_regressed_mask", phase_regressed_mask)
        rewards = torch.zeros((num_envs, num_phases), device=device)


        dist_p1_ee = torch.norm(p1_pos - ee_1, dim=-1)
        
        rewards[:, Phases.REACH_P1.value] = torch.where(
                            self.not_visited_mask[:, Phases.REACH_P1.value],
                            2 * torch.exp(-50 * dist_p1_ee) ,
                            rewards[:, Phases.REACH_P1.value]  # leave existing reward unchanged
                        )
        #rewards[:, Phases.REACH_P1.value] = 2 * torch.exp(-50 * dist_p1_ee) 
        env_ids = torch.arange(self.num_envs)
        mask = (dist_p1_ee <= self.phase_detector.CLOSE_THRESHOLD) & (self.not_visited_mask[env_ids, Phases.REACH_P1.value] 
                    & (phases_one_hot[env_ids, Phases.REACH_OBJ.value].bool()))

        #print("visited", self.not_visited_mask, mask, self.not_visited_mask[env_ids, Phases.REACH_P1.value])
        self.not_visited_mask[mask, Phases.REACH_P1.value] = False
        rewards[mask, Phases.REACH_OBJ.value] += 2000
        #print("rew", rewards)
        # phase 0: REACH_OBJ
        dist_obj_ee = torch.norm(obj_pos - ee_1, dim=-1)
        dist_obj_griplnk = torch.norm(obj_griplink_pos - gripper_link_pos, dim=-1)
        rewards[:, Phases.REACH_OBJ.value] += torch.where(
                            self.not_visited_mask[:, Phases.REACH_OBJ.value],
                            2* torch.exp(-50 * dist_obj_ee) ,
                            rewards[:, Phases.REACH_OBJ.value] )
        rewards[:, Phases.REACH_OBJ.value] += torch.where(
                            self.not_visited_mask[:, Phases.REACH_OBJ.value],
                            2* torch.exp(-50 * dist_obj_griplnk) ,
                            rewards[:, Phases.REACH_OBJ.value] )
        # rewards[:, Phases.REACH_OBJ.value] +=  (
        #     2* torch.exp(-50 * dist_obj_ee)
        # )
        

        # print("dist obj ee", dist)
        
        # phase 1: GRIP_1_OPEN
        gripper_width = self.phase_detector.get_gripper_width(self.robot_1)
        log_if(not self.cfg.is_training, "gripper width", gripper_width)
        #
        
        mask_ro = ((dist_obj_ee <= self.phase_detector.SUPER_CLOSE_THRESHOLD) & 
                (dist_obj_griplnk <= self.phase_detector.SUPER_CLOSE_THRESHOLD)) &  (
                    self.not_visited_mask[env_ids, Phases.REACH_OBJ.value] & 
                    (phases_one_hot[env_ids, Phases.GRIP_1_OPEN.value].bool()))

        
        self.not_visited_mask[mask_ro, Phases.REACH_OBJ.value] = False
        rewards[mask_ro, Phases.GRIP_1_OPEN.value] += 2000

        rewards[:, Phases.GRIP_1_OPEN.value] += torch.where(
                            self.not_visited_mask[:, Phases.GRIP_1_OPEN.value],
                            gripper_width * 20 ,
                            rewards[:, Phases.GRIP_1_OPEN.value] )
        #rewards[:, Phases.GRIP_1_OPEN.value] +=  gripper_width * 2

        mask_open = (dist_obj_ee <= self.phase_detector.SUPER_CLOSE_THRESHOLD) & (
                    ~self.phase_detector.is_gripper_closed(self.robot_1) & 
                    self.not_visited_mask[env_ids, Phases.GRIP_1_OPEN.value] & 
                    (phases_one_hot[env_ids, Phases.REACH_OBJ_GRIP.value].bool()))

        
        self.not_visited_mask[mask_open, Phases.GRIP_1_OPEN.value] = False
        rewards[mask_open, Phases.REACH_OBJ_GRIP.value] += 2000

        dist_grp_tgt = self.get_grp_tgt_distance(self.robot_1)
        dist_grp1_tgt, dist_grp2_tgt = dist_grp_tgt
        rewards[:, Phases.REACH_OBJ_GRIP.value] += torch.where(
                            self.not_visited_mask[:, Phases.REACH_OBJ_GRIP.value],
                            2* torch.exp(-100 * dist_grp1_tgt) ,
                            rewards[:, Phases.REACH_OBJ_GRIP.value] )
        
        rewards[:, Phases.REACH_OBJ_GRIP.value] += torch.where(
                            self.not_visited_mask[:, Phases.REACH_OBJ_GRIP.value],
                            2* torch.exp(-100 * dist_grp2_tgt) ,
                            rewards[:, Phases.REACH_OBJ_GRIP.value] )
        
        # dist_gripper_tgt = torch.norm(gripper_link_tgt - gripper_link_pos, dim=-1)
        # rewards[:, Phases.REACH_OBJ_GRIP.value] += torch.where(
        #                     self.not_visited_mask[:, Phases.REACH_OBJ_GRIP.value],
        #                     2* torch.exp(-100 * dist_gripper_tgt) ,
        #                     rewards[:, Phases.REACH_OBJ_GRIP.value] )

        #rewards[:, Phases.REACH_OBJ_GRIP.value] += 2* torch.exp(-100 * dist_obj_grip_ee1)



        mask_grip = ((dist_grp1_tgt <= self.phase_detector.GRIP_CLOSE_THRESHOLD) & 
                     (dist_grp2_tgt <= self.phase_detector.GRIP_CLOSE_THRESHOLD) 
                     #(dist_gripper_tgt <= self.phase_detector.GRIP_CLOSE_THRESHOLD)
                     ) & ( 
                    self.not_visited_mask[env_ids, Phases.REACH_OBJ_GRIP.value] & 
                    (phases_one_hot[env_ids, Phases.GRIP_1_CLOSE.value].bool()))

        
        self.not_visited_mask[mask_grip, Phases.REACH_OBJ_GRIP.value] = False
        rewards[mask_grip, Phases.GRIP_1_CLOSE.value] += 2000
        # phase 2: GRIP_1_CLOSE

        rewards[:, Phases.GRIP_1_CLOSE.value] += torch.where(
                            self.not_visited_mask[:, Phases.GRIP_1_CLOSE.value],
                            200 * torch.exp(-10 * gripper_width) ,
                            rewards[:, Phases.GRIP_1_CLOSE.value] )
        #rewards[:, Phases.GRIP_1_CLOSE.value] += 200* torch.exp(-5 * gripper_width)


        mask_close = (gripper_width < 0.15) & (
                        self.not_visited_mask[env_ids, Phases.GRIP_1_CLOSE.value] 
                            & ~self.not_visited_mask[env_ids, Phases.REACH_OBJ.value]) & (
                                phases_one_hot[env_ids, Phases.LIFT.value].bool()
                            )

        
        self.not_visited_mask[mask_close, Phases.GRIP_1_CLOSE.value] = False
        rewards[mask_close, Phases.LIFT.value] += 2000

        # phase 3: LIFT
        height = obj_abs_pos[:, 2] - self.cfg.ground_height
        rewards[:, Phases.LIFT.value] += torch.where(
                            self.not_visited_mask[:, Phases.LIFT.value],
                            2*height ,
                            rewards[:, Phases.LIFT.value] )
        #rewards[:, Phases.LIFT.value] += 10*height

        log_if(not self.cfg.is_training, f"obj_pos z {obj_abs_pos[:, 2]} height {height}")

        # phase 4: REACH_GOAL_1
        mask_reach1 = self.phase_detector.is_object_above_ground() & (
                        self.not_visited_mask[env_ids, Phases.LIFT.value] 
                            & ~self.not_visited_mask[env_ids, Phases.GRIP_1_CLOSE.value]) & (
                                phases_one_hot[env_ids, Phases.REACH_GOAL_1.value].bool()
                            )

        
        self.not_visited_mask[mask_reach1, Phases.LIFT.value] = False
        rewards[mask_reach1, Phases.REACH_GOAL_1.value] += 2000

        dist_goal1 = torch.norm(goal_pos - ee_1, dim=-1)
        rewards[:, Phases.REACH_GOAL_1.value] += 2 * torch.exp(-50 * dist_goal1)

        # phase 5: REACH_GOAL_2
        dist_goal2 = torch.norm(goal_pos - ee_2, dim=-1)
        rewards[:, Phases.REACH_GOAL_2.value] = torch.exp(-self.cfg.dist_reward_scale * dist_goal2)

        # phase 6: GRIP_2
        holding_2 = self.phase_detector.is_holding_object(self.robot_2)
        rewards[:, Phases.GRIP_2.value] = 10 * holding_2.float()

        # phase 7: RELEASE_1
        holding_1 = self.phase_detector.is_holding_object(self.robot_1)
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
        log_if(not self.cfg.is_training, "visited", self.not_visited_mask)
        log_if(not self.cfg.is_training, f"rewards {final_rewards_r1} rewards_all {rewards}")
        
        return {
            "robot_1": final_rewards_r1,
            "robot_2": final_rewards_r2
        }

    def _get_dones(self):
        self._compute_intermediate_values()
        obj_pos_z = self.object.data.root_pos_w[:, 2]
        fallen = obj_pos_z < self.cfg.fall_z_threshold
        timeout = self.episode_length_buf >= self.max_episode_length - 1



        # if ee reaches between p1 and obj before it reaches p1, stop the env
        ee1_pos = self._get_ee_position(self.robot_1)
        obj_pos = self._get_obj_pos()
        p1_pos = self.get_p1_pos(obj_pos)
        # if ee has not visited p1 but is inbetween p1 and obj end the env
        invalid_move = self.is_point_between_parallel_lines_done(obj_pos, p1_pos, ee1_pos) & self.not_visited_mask[:, Phases.REACH_P1.value]
        
        # for now i am setting this for both, because i want to be sure env ends
        #print("invalid move", self.is_point_between_parallel_lines_done(obj_pos, p1_pos, ee1_pos, ), self.not_visited_mask[:, Phases.REACH_P1.value])
        # if self.common_step_counter > 30000:
        #     terminated = fallen | invalid_move
        # else:
        terminated = fallen #| self.phase_regressed_mask
        terminated = torch.zeros_like(timeout, dtype=torch.bool)

        return (
            {agent: terminated.clone()  for agent in self.cfg.possible_agents},
            {agent: timeout.clone() for agent in self.cfg.possible_agents}
        )

    

    def _reset_idx(self, env_ids):
        
        super()._reset_idx(env_ids)
       
        self.phase_regressed_mask = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # Set to True — all phases are not visited initially
        self.not_visited_mask = torch.ones((self.num_envs, len(Phases)), dtype=torch.bool, device=self.device)
        # In __init__ or reset()
        self.phase_visit_counts = torch.zeros((self.num_envs, len(Phases)), device=self.device, dtype=torch.float)

        # Reset object pose with some noise

        x_noise = sample_uniform(0, 0.05, (len(env_ids), 1), self.device)
        y_noise = sample_uniform(-0.05, 0.05, (len(env_ids), 1), self.device)
        z_noise = sample_uniform(0, 0.01, (len(env_ids), 1), self.device)

        pos_noise = torch.cat([x_noise, y_noise, z_noise], dim=1)
        rot_noise = self.cfg.reset_rot_noise * sample_uniform(-1, 1, (len(env_ids), 2), self.device)
        
        new_pos = self.scene.env_origins[env_ids] + pos_noise
        
        new_rot = randomize_rotation(rot_noise[:, 0], rot_noise[:, 1])
        # new_rot[0] = torch.tensor([0.7071, 0, 0, 0.7071])
        new_rot = torch.tensor([0.7071, 0, 0, -0.7071], device=self.device).unsqueeze(0).repeat(len(env_ids), 1)
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
        self.robot_1_prev_deltas = torch.zeros_like(self.robot_1.data.joint_pos[:, self.actuated_dof_indices])
        self.robot_2_prev_deltas = torch.zeros_like(self.robot_2.data.joint_pos[:, self.actuated_dof_indices])

        
        
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

