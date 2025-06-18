import torch
import numpy as np
from typing import Dict, Any
from enum import Enum
import torch.nn.functional as F

class Phases(Enum):

    # Until reaching the object
    REACH_OBJ = 0
    # R1 grips 1
    # until r1 opens the gripper
    GRIP_1_OPEN = 1

    GRIP_1_CLOSE = 2
    # until r1 lifts the obj
    LIFT = 3
    # R1 reaches goal 1
    REACH_GOAL_1 = 4
    # R2 reaches near ee of R1
    REACH_GOAL_2 = 5
    #R2 grips object
    GRIP_2 = 6
    #R1 releases object
    RELEASE_1 = 7
    # Task completed
    END = 8

class PhaseDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        # Distance thresholds for phase detection
        self.CLOSE_THRESHOLD = 0.05  # 5cm
        self.SUPER_CLOSE_THRESHOLD = 0.01  # 2cm
        self.FAR_THRESHOLD = 0.15  # 15cm
        self.GRIP_THRESHOLD = 0.015
        self.GRIP_CLOSE_THRESHOLD = 0.009
        
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
    
    # def get_phases(self, agents, batch_size, obj_position, goal_position, prev_phases) -> torch.Tensor:
    #     """
    #     Detect current phase for each environment
    #     Returns tensor of Phases enum values for each environment
    #     """
        
    #     phases = [Phases.REACH_OBJ] * batch_size  # Initialize with default phase
        
    #     # # # Get robot references
    #     robot_1, robot_2 = agents
        
    #     # Get positions
    #     ee_1_pos = self._get_ee_position(robot_1)
    #     ee_2_pos = self._get_ee_position(robot_2)
        
    #     # Get object position (assuming you have this in your environment)
        
        
    #     # Get gripper states
    #     gripper_1_closed = self.is_gripper_closed(robot_1)
    #     gripper_2_closed = self.is_gripper_closed(robot_2)
        
    #     # Check object holding states
    #     robot_1_holding = self._is_holding_object(obj_position, robot_1)
    #     robot_2_holding = self._is_holding_object(obj_position, robot_2)
        
    #     # Check if object is above ground
    #     obj_above_ground = self.is_object_above_ground(obj_position)
        
    #     # Calculate distances
    #     ee1_obj_dist = self._get_distance(ee_1_pos, obj_position)
    #     ee1_goal_dist = self._get_distance(ee_1_pos, goal_position)
    #     ee2_goal_dist = self._get_distance(ee_2_pos, goal_position)
       
    #     # Phase detection logic
    #     for i in range(batch_size):
    #         # REACH_OBJ: ee_1 is far away from obj and obj is in ground
    #         if (ee1_obj_dist[i] > self.FAR_THRESHOLD and 
    #             not obj_above_ground[i]):# and prev_phases[i] == Phases.REACH_OBJ):
    #             phases[i] = Phases.REACH_OBJ
                
    #         # GRIP_1: ee_1 is close to obj, obj is in ground, finger_1 is closed
    #         elif (ee1_obj_dist[i] <= self.GRIP_THRESHOLD and 
    #               not obj_above_ground[i]):# and
    #               #( prev_phases[i] == Phases.REACH_OBJ or prev_phases[i] == Phases.GRIP_1_OPEN)):
    #             phases[i] = Phases.GRIP_1_OPEN

            
    #         elif (ee1_obj_dist[i] <= self.GRIP_THRESHOLD and 
    #               not obj_above_ground[i] and not gripper_1_closed[i]):
    #         #       and (prev_phases[i] == Phases.GRIP_1_OPEN or prev_phases[i]== Phases.GRIP_1_CLOSE)):
    #             phases[i] = Phases.GRIP_1_CLOSE
                
    #         # LIFT: ee_1 is super close to obj and obj is in ground, gripper_1 is open
    #         elif (ee1_obj_dist[i] <= self.GRIP_THRESHOLD and 
    #               not obj_above_ground[i] and gripper_1_closed[i] 
    #               and (prev_phases[i] == Phases.GRIP_1_CLOSE or prev_phases[i]== Phases.LIFT)):
    #             phases[i] = Phases.LIFT
                
    #         # REACH_GOAL_1: gripper_1 is holding obj, obj is above ground, gripper_1 closed, ee1 far away from goal_pos
    #         elif (robot_1_holding[i] and 
    #               obj_above_ground[i] and 
                  
    #               ee1_goal_dist[i] > self.FAR_THRESHOLD):
    #             phases[i] = Phases.REACH_GOAL_1
                
    #         # REACH_GOAL_2: gripper_1 is holding obj, obj is above ground, gripper_1 closed, ee1 close to goal_pos and ee_2 is far away from goal_position
    #         elif (robot_1_holding[i] and 
    #               obj_above_ground[i] and 
    #               gripper_1_closed[i] and 
    #               ee1_goal_dist[i] <= self.CLOSE_THRESHOLD and 
    #               ee2_goal_dist[i] > self.FAR_THRESHOLD):
    #             phases[i] = Phases.REACH_GOAL_2
                
    #         # GRIP_2: gripper_1 is holding obj, obj is above ground, gripper_1 closed, ee1 close to goal_pos and ee_2 is near goal and gripper_2 is closed and gripper_2 not holding obj
    #         elif (robot_1_holding[i] and 
    #               obj_above_ground[i] and 
    #               gripper_1_closed[i] and 
    #               ee1_goal_dist[i] <= self.CLOSE_THRESHOLD and 
    #               ee2_goal_dist[i] <= self.CLOSE_THRESHOLD and 
    #               gripper_2_closed[i] and 
    #               not robot_2_holding[i]):
    #             phases[i] = Phases.GRIP_2
                
    #         # RELEASE_1: gripper_1 is holding obj, obj is above ground, gripper_1 closed, ee1 close to goal_pos and ee_2 is near goal and gripper_2 is closed and gripper_2 is holding obj
    #         elif (robot_1_holding[i] and 
    #               obj_above_ground[i] and 
    #               gripper_1_closed[i] and 
    #               ee1_goal_dist[i] <= self.CLOSE_THRESHOLD and 
    #               ee2_goal_dist[i] <= self.CLOSE_THRESHOLD and 
    #               gripper_2_closed[i] and 
    #               robot_2_holding[i]):
    #             phases[i] = Phases.RELEASE_1
                
    #         # END: gripper_1 is not holding obj, obj is above ground, ee1 close to goal_pos and ee_2 is near goal and gripper_2 is closed and gripper_2 is holding obj
    #         elif (not robot_1_holding[i] and 
    #               obj_above_ground[i] and 
    #               ee1_goal_dist[i] <= self.CLOSE_THRESHOLD and 
    #               ee2_goal_dist[i] <= self.CLOSE_THRESHOLD and 
    #               gripper_2_closed[i] and 
    #               robot_2_holding[i]):
    #             phases[i] = Phases.END
        
    #     return phases

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

        ee1_obj_dist = self._get_distance(ee_1_pos, obj_position)
        ee1_goal_dist = self._get_distance(ee_1_pos, goal_position)
        ee2_goal_dist = self._get_distance(ee_2_pos, goal_position)

        num_envs = batch_size
        num_phases = len(Phases)
        device = ee_1_pos.device

        # initialize all phases to False
        phase_mask = torch.zeros((num_envs, num_phases), dtype=torch.bool, device=device)

        # PHASE 0: REACH_OBJ
        phase_mask[:, Phases.REACH_OBJ.value] = (
            (ee1_obj_dist > self.FAR_THRESHOLD) &
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

    # For environments with True phases, find the maximum index where True
        if valid_mask.any():
            # Get the last True index for each row
            flipped_mask = phase_mask.flip(dims=[1])  # Flip to make last index first
            first_true_in_flipped = torch.argmax(flipped_mask.int(), dim=1)  # Get first True in flipped
            last_true_in_original = num_phases - 1 - first_true_in_flipped  # Convert back to original indexing
            phase_indices = torch.where(valid_mask, last_true_in_original, phase_indices)
        one_hot = F.one_hot(phase_indices, num_classes=num_phases).float()  # (num_envs, num_phases)

        return one_hot
