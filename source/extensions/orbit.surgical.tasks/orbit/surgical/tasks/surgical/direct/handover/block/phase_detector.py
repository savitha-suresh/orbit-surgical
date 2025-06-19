import torch
import numpy as np
from typing import Dict, Any
from enum import Enum
import torch.nn.functional as F

class Phases(Enum):


    REACH_P1 = 0
    # Until reaching the object
    REACH_OBJ = 1
    # R1 grips 1
    # until r1 opens the gripper
    GRIP_1_OPEN = 2

    GRIP_1_CLOSE = 3
    # until r1 lifts the obj
    LIFT = 4
    # R1 reaches goal 1
    REACH_GOAL_1 = 5
    # R2 reaches near ee of R1
    REACH_GOAL_2 = 6
    #R2 grips object
    GRIP_2 = 7
    #R1 releases object
    RELEASE_1 = 8
    # Task completed
    END = 9

class PhaseDetector:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        # Distance thresholds for phase detection
        self.CLOSE_THRESHOLD = 0.05  # 5cm
        self.SUPER_CLOSE_THRESHOLD = 0.01  # 2cm
        self.FAR_THRESHOLD = 0.15  # 15cm
        self.GRIP_THRESHOLD = 0.05
        self.GRIP_CLOSE_THRESHOLD = 0.05
        
    def _get_ee_position(self, robot):
        ee_pos = robot.data.body_pos_w[:, robot.find_bodies(self.cfg.ee_link_name)[0]]
        return ee_pos.squeeze(1)
    

    def get_gripper_pos(self, robot):
        gripper_joints = self.cfg.finger_joint_names
        joint_pos = robot.data.joint_pos
        
        # Get joint indices
        gripper_indices = []
        for joint_name in gripper_joints:
            joint_idx = robot.find_joints(joint_name)[0]
            gripper_indices.append(joint_idx)
        
        # Check if gripper is closed (small joint values)
        gripper_pos = joint_pos[:, gripper_indices]
        
        gripper_pos = gripper_pos.squeeze(2)
        return gripper_pos
    

    def get_gripper_width(self, robot):
        gripper_pos = self.get_gripper_pos(robot)
        gripper_distance = torch.abs(gripper_pos[:, 0]) + torch.abs(gripper_pos[:, 1])
        return gripper_distance

    
    
    def is_gripper_closed(self, robot):
        """Get gripper state (open/closed) based on joint positions"""
        
        
        # Get gripper joint positions
        gripper_pos = self.get_gripper_pos(robot)
        # Gripper is closed if both joints are close to closed position
        closed_threshold = 0.05  # Adjust based on your gripper
        is_closed = torch.all(torch.abs(gripper_pos) < closed_threshold, dim=-1)
        
        return is_closed
    
    def is_holding_object(self, obj_position, robot):
        return self._is_holding_object(obj_position, robot)
    
    def _is_holding_object(self, obj_position, robot):
        """Check if robot is holding the object"""
        
        ee_position = self._get_ee_position(robot)
        gripper_closed = self.is_gripper_closed(robot)
        
        # Object is being held if gripper is closed and EE is very close to object
        ee_obj_distance = torch.norm(ee_position - obj_position, dim=-1)
        is_holding = gripper_closed & (ee_obj_distance < self.SUPER_CLOSE_THRESHOLD)
        
        return is_holding
    
    
    def is_object_above_ground(self, obj_position, ground_height=0.0149):
        """Check if object is above ground"""
        ground_height = self.cfg.ground_height
        return obj_position[:, 2] > (ground_height + 0.05)  # 1cm above ground
    
    def _get_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        return torch.norm(pos1 - pos2, dim=-1)
    
    
    def get_phases(self, agents, batch_size, obj_position, goal_position, prev_phases):
        robot_1, robot_2 = agents

        ee_1_pos = self._get_ee_position(robot_1)         # (num_envs, 3)
        ee_2_pos = self._get_ee_position(robot_2)         # (num_envs, 3)
        prev_phases = prev_phases.bool()
        #print("prev_phases", prev_phases)
        gripper_1_closed = self.is_gripper_closed(robot_1)  # (num_envs,)
        gripper_2_closed = self.is_gripper_closed(robot_2)  # (num_envs,)
        robot_1_holding = self._is_holding_object(obj_position, robot_1)  # (num_envs,)
        robot_2_holding = self._is_holding_object(obj_position, robot_2)  # (num_envs,)

        obj_above_ground = self.is_object_above_ground(obj_position)      # (num_envs,)
        p1_pos = self.env.get_p1_pos()
        ee1_p1_dist = self._get_distance(ee_1_pos, p1_pos)
        ee1_obj_dist = self._get_distance(ee_1_pos, obj_position)
        ee1_goal_dist = self._get_distance(ee_1_pos, goal_position)
        ee2_goal_dist = self._get_distance(ee_2_pos, goal_position)

        num_envs = batch_size
        num_phases = len(Phases)
        device = ee_1_pos.device

        # initialize all phases to False
        phase_mask = torch.zeros((num_envs, num_phases), dtype=torch.bool, device=device)
        # PHASE 0: REACH_OBJ
        

        phase_mask[:, Phases.REACH_P1.value] = (
            (ee1_p1_dist >= self.FAR_THRESHOLD) 
        )

        phase_mask[:, Phases.REACH_OBJ.value] = (
            (ee1_obj_dist > self.GRIP_THRESHOLD) &
            (~obj_above_ground)
        )

        # PHASE 1: GRIP_1_OPEN
        phase_mask[:, Phases.GRIP_1_OPEN.value] = (
            (ee1_obj_dist <= self.GRIP_THRESHOLD) &
            (~obj_above_ground) & ( prev_phases[:, Phases.REACH_OBJ.value] | prev_phases[:, Phases.GRIP_1_OPEN.value])
        )

        # PHASE 2: GRIP_1_CLOSE
        phase_mask[:, Phases.GRIP_1_CLOSE.value] = (
            (ee1_obj_dist <= self.GRIP_CLOSE_THRESHOLD) &
            (~obj_above_ground) &
            (~gripper_1_closed) & (prev_phases[:, Phases.GRIP_1_OPEN.value] | prev_phases[:, Phases.GRIP_1_CLOSE.value])
        )

        # PHASE 3: LIFT
        phase_mask[:, Phases.LIFT.value] = (
            (ee1_obj_dist <= self.GRIP_CLOSE_THRESHOLD) &
            (~obj_above_ground) &
            gripper_1_closed & (prev_phases[:, Phases.GRIP_1_CLOSE.value] | prev_phases[:, Phases.LIFT.value])
        )

        # PHASE 4: REACH_GOAL_1
        phase_mask[:, Phases.REACH_GOAL_1.value] = (
            robot_1_holding &
            obj_above_ground &
            (ee1_goal_dist > self.FAR_THRESHOLD)
        )

        # PHASE 5: REACH_GOAL_2
        phase_mask[:, Phases.REACH_GOAL_2.value] = (
            robot_1_holding &
            obj_above_ground &
            gripper_1_closed &
            (ee1_goal_dist <= self.CLOSE_THRESHOLD) &
            (ee2_goal_dist > self.FAR_THRESHOLD)
        )

        # PHASE 6: GRIP_2
        phase_mask[:, Phases.GRIP_2.value] = (
            robot_1_holding &
            obj_above_ground &
            gripper_1_closed &
            (ee1_goal_dist <= self.CLOSE_THRESHOLD) &
            (ee2_goal_dist <= self.CLOSE_THRESHOLD) &
            gripper_2_closed &
            (~robot_2_holding)
        )

        # PHASE 7: RELEASE_1
        phase_mask[:, Phases.RELEASE_1.value] = (
            robot_1_holding &
            obj_above_ground &
            gripper_1_closed &
            (ee1_goal_dist <= self.CLOSE_THRESHOLD) &
            (ee2_goal_dist <= self.CLOSE_THRESHOLD) &
            gripper_2_closed &
            robot_2_holding
        )

        # PHASE 8: END
        phase_mask[:, Phases.END.value] = (
            (~robot_1_holding) &
            obj_above_ground &
            (ee1_goal_dist <= self.CLOSE_THRESHOLD) &
            (ee2_goal_dist <= self.CLOSE_THRESHOLD) &
            gripper_2_closed &
            robot_2_holding
        )
        
        # priority logic: use highest index where True
        # reversed_mask = phase_mask.flip(dims=[1])
        # reversed_indices = torch.argmax(reversed_mask.int(), dim=1)
        # last_indices = phase_mask.size(1) - 1 - reversed_indices
        valid_mask = phase_mask.any(dim=1)  # (num_envs,) - which envs have any True phase
        phase_indices = torch.zeros(num_envs, dtype=torch.long, device=device)
        #print(phase_mask)
    # For environments with True phases, find the maximum index where True
        if valid_mask.any():
            # Get the last True index for each row
            flipped_mask = phase_mask.flip(dims=[1])  # Flip to make last index first
            first_true_in_flipped = torch.argmax(flipped_mask.int(), dim=1)  # Get first True in flipped
            last_true_in_original = num_phases - 1 - first_true_in_flipped  # Convert back to original indexing
            phase_indices = torch.where(valid_mask, last_true_in_original, phase_indices)


        prev_phase_indices = torch.argmax(prev_phases.int(), dim=1)
        phase_regressed_mask = (phase_indices < prev_phase_indices)
        phase_same_mask = (phase_indices == prev_phase_indices)
        one_hot = F.one_hot(phase_indices, num_classes=num_phases).float()  # (num_envs, num_phases)
        
        return one_hot, phase_indices, phase_regressed_mask, phase_same_mask
