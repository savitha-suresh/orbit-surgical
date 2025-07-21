# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Enhanced evaluation script for RL agent from skrl with comprehensive metrics collection.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import json
import os
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent from skrl with metrics collection.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# Evaluation specific arguments
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate.")
parser.add_argument("--height_threshold", type=float, default=0.1, help="Height threshold for object success.")
parser.add_argument("--results_dir", type=str, default="evaluation_results", help="Directory to save results.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import orbit.surgical.tasks
# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
algorithm = args_cli.algorithm.lower()


# class EvaluationMetrics:
#     """Class to handle evaluation metrics collection and analysis."""
    
#     def __init__(self, num_envs: int, num_phases: int = 6, height_threshold: float = 0.1):
#         self.num_envs = num_envs
#         self.num_phases = num_phases
#         self.height_threshold = height_threshold
#         self.reset_episode_data()
        
#         # Store results across all episodes
#         self.all_episodes_data = []
        
#     def reset_episode_data(self):
#         """Reset data for a new episode."""
#         self.episode_data = {
#             'phase_reached_count': np.zeros(self.num_phases),  # Count of envs reaching each phase
#             'phase_reached_timesteps': [[] for _ in range(self.num_phases)],  # Timesteps when each phase was reached
#             'initial_obj_positions': np.zeros((self.num_envs, 3)),  # Initial object positions for each env
#             'final_obj_positions': np.zeros((self.num_envs, 3)),  # Final object positions for each env
#             'height_success_timesteps': [],  # Timesteps when height threshold was reached
#             'height_success_count': 0,  # Number of envs that reached height threshold
#             'episode_lengths': np.zeros(self.num_envs),  # Episode length for each env
#             'episode_rewards': np.zeros(self.num_envs),  # Final reward for each env
#             'total_timesteps': 0
#         }
        
#     def store_initial_positions(self, initial_obj_positions: np.ndarray):
#         """Store initial object positions for the episode.
        
#         Args:
#             initial_obj_positions: Shape (num_envs, 3) - Initial object positions
#         """
#         self.episode_data['initial_obj_positions'] = initial_obj_positions.copy()
    
#     def get_position_std(self, positions: np.ndarray) -> Dict[str, float]:
#         """Calculate standard deviation of x, y, z coordinates.
        
#         Args:
#             positions: Shape (num_envs, 3) - Object positions
            
#         Returns:
#             Dictionary with std for x, y, z coordinates
#         """
#         return {
#             'x_std': float(np.std(positions[:, 0])),
#             'y_std': float(np.std(positions[:, 1])),
#             'z_std': float(np.std(positions[:, 2]))
#         }
    
#     def update_phase_data(self, not_visited_mask: np.ndarray, timestep: int):
#         """Update phase data based on not_visited_mask.
        
#         Args:
#             not_visited_mask: Shape (num_envs, num_phases) - False when phase is reached
#             timestep: Current timestep
#         """
#         # Check which phases were just reached (became False)
#         for phase_idx in range(self.num_phases):
#             newly_reached = ~not_visited_mask[:, phase_idx]  # True where phase was reached
            
#             for env_idx in range(self.num_envs):
#                 if newly_reached[env_idx]:
#                     # Check if this is the first time this env reached this phase
#                     if timestep not in self.episode_data['phase_reached_timesteps'][phase_idx]:
#                         self.episode_data['phase_reached_count'][phase_idx] += 1
#                         self.episode_data['phase_reached_timesteps'][phase_idx].append(timestep)
    
#     def update_height_data(self, obj_positions: np.ndarray, timestep: int):
#         """Update height-related data.
        
#         Args:
#             obj_positions: Shape (num_envs, 3) - Object positions
#             timestep: Current timestep
#         """
#         # Check which objects are above height threshold
#         above_threshold = obj_positions[:, 2] > self.height_threshold  # Assuming Z is height
        
#         for env_idx in range(self.num_envs):
#             if above_threshold[env_idx]:
#                 # Record the timestep when height threshold was first reached
#                 if timestep not in self.episode_data['height_success_timesteps']:
#                     self.episode_data['height_success_timesteps'].append(timestep)
#                     self.episode_data['height_success_count'] += 1
    
#     def finalize_episode(self, final_obj_positions: np.ndarray, episode_lengths: np.ndarray, 
#                         episode_rewards: np.ndarray, total_timesteps: int):
#         """Finalize episode data and store results."""
#         self.episode_data['final_obj_positions'] = final_obj_positions.copy()
#         self.episode_data['episode_lengths'] = episode_lengths.copy()
#         self.episode_data['episode_rewards'] = episode_rewards.copy()
#         self.episode_data['total_timesteps'] = total_timesteps
        
#         # Store episode data
#         self.all_episodes_data.append(self.episode_data.copy())
        
#     def get_episode_summary(self, episode_idx: int) -> Dict:
#         """Get summary statistics for a specific episode."""
#         if episode_idx >= len(self.all_episodes_data):
#             return {}
            
#         data = self.all_episodes_data[episode_idx]
        
#         # Calculate standard deviations for initial and final positions
#         initial_position_std = self.get_position_std(data['initial_obj_positions'])
#         final_position_std = self.get_position_std(data['final_obj_positions'])
        
#         summary = {
#             'episode': episode_idx,
#             'total_timesteps': data['total_timesteps'],
#             'phase_success_rates': {
#                 f'phase_{i+1}': float(data['phase_reached_count'][i]) / self.num_envs 
#                 for i in range(self.num_phases)
#             },
#             'height_success_rate': float(data['height_success_count']) / self.num_envs,
#             'avg_episode_length': float(np.mean(data['episode_lengths'])),
#             'avg_episode_reward': float(np.mean(data['episode_rewards'])),
#             'initial_position_std': initial_position_std,
#             'final_position_std': final_position_std
#         }
        
#         return summary
    
#     def get_overall_summary(self) -> Dict:
#         """Get overall summary statistics across all episodes."""
#         if not self.all_episodes_data:
#             return {}
            
#         num_episodes = len(self.all_episodes_data)
        
#         # Aggregate phase success rates
#         phase_success_rates = {}
#         for phase_idx in range(self.num_phases):
#             phase_successes = [data['phase_reached_count'][phase_idx] for data in self.all_episodes_data]
#             phase_success_rates[f'phase_{phase_idx+1}'] = {
#                 'mean': float(np.mean(phase_successes)) / self.num_envs,
#                 'std': float(np.std(phase_successes)) / self.num_envs,
#                 'min': float(np.min(phase_successes)) / self.num_envs,
#                 'max': float(np.max(phase_successes)) / self.num_envs
#             }
        
#         # Aggregate height success rates
#         height_successes = [data['height_success_count'] for data in self.all_episodes_data]
#         height_success_rate = {
#             'mean': float(np.mean(height_successes)) / self.num_envs,
#             'std': float(np.std(height_successes)) / self.num_envs,
#             'min': float(np.min(height_successes)) / self.num_envs,
#             'max': float(np.max(height_successes)) / self.num_envs
#         }
        
#         # Aggregate episode lengths
#         all_episode_lengths = np.concatenate([data['episode_lengths'] for data in self.all_episodes_data])
#         episode_length_stats = {
#             'mean': float(np.mean(all_episode_lengths)),
#             'std': float(np.std(all_episode_lengths)),
#             'min': float(np.min(all_episode_lengths)),
#             'max': float(np.max(all_episode_lengths))
#         }
        
#         # Aggregate rewards
#         all_rewards = np.concatenate([data['episode_rewards'] for data in self.all_episodes_data])
#         reward_stats = {
#             'mean': float(np.mean(all_rewards)),
#             'std': float(np.std(all_rewards)),
#             'min': float(np.min(all_rewards)),
#             'max': float(np.max(all_rewards))
#         }
        
#         # Calculate overall position standard deviations
#         all_initial_positions = np.concatenate([data['initial_obj_positions'] for data in self.all_episodes_data])
#         all_final_positions = np.concatenate([data['final_obj_positions'] for data in self.all_episodes_data])
        
#         initial_position_std = self.get_position_std(all_initial_positions)
#         final_position_std = self.get_position_std(all_final_positions)
        
#         summary = {
#             'num_episodes': num_episodes,
#             'num_envs_per_episode': self.num_envs,
#             'total_trials': num_episodes * self.num_envs,
#             'phase_success_rates': phase_success_rates,
#             'height_success_rate': height_success_rate,
#             'episode_length_stats': episode_length_stats,
#             'reward_stats': reward_stats,
#             'initial_position_std': initial_position_std,
#             'final_position_std': final_position_std
#         }
        
#         return summary
    
#     def save_results(self, results_dir: str, task_name: str):
#         """Save all results to files."""
#         os.makedirs(results_dir, exist_ok=True)
        
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         base_filename = f"{task_name}_{timestamp}"
        
#         # Save overall summary
#         overall_summary = self.get_overall_summary()
#         with open(os.path.join(results_dir, f"{base_filename}_summary.json"), 'w') as f:
#             json.dump(overall_summary, f, indent=2)
        
#         # Save episode-by-episode data
#         episode_summaries = []
#         for i in range(len(self.all_episodes_data)):
#             episode_summaries.append(self.get_episode_summary(i))
        
#         with open(os.path.join(results_dir, f"{base_filename}_episodes.json"), 'w') as f:
#             json.dump(episode_summaries, f, indent=2)
        
#         # Save raw data
#         with open(os.path.join(results_dir, f"{base_filename}_raw.json"), 'w') as f:
#             # Convert numpy arrays to lists for JSON serialization
#             serializable_data = []
#             for episode_data in self.all_episodes_data:
#                 serializable_episode = {}
#                 for key, value in episode_data.items():
#                     if isinstance(value, np.ndarray):
#                         serializable_episode[key] = value.tolist()
#                     else:
#                         serializable_episode[key] = value
#                 serializable_data.append(serializable_episode)
#             json.dump(serializable_data, f, indent=2)
        
#         print(f"Results saved to {results_dir}")
#         print(f"Summary: {base_filename}_summary.json")
#         print(f"Episodes: {base_filename}_episodes.json")
#         print(f"Raw data: {base_filename}_raw.json")

class EvaluationMetrics:
    def __init__(self, num_envs: int, num_phases: int = 6, height_threshold: float = 0.1, num_episodes = 10):
        self.num_envs = num_envs
        self.num_phases = num_phases
        self.height_threshold = height_threshold
        self.reset_episode_data()
        self.all_episodes_data = []
        self.num_episodes = num_episodes

    def reset_episode_data(self):
        self.episode_data = {
            'phase_reached_count': torch.zeros(self.num_phases),
            'phase_reached_timesteps': [[] for _ in range(self.num_phases)],
            'initial_obj_positions': torch.zeros((self.num_envs, 3)),
            'final_obj_positions': torch.zeros((self.num_envs, 3)),
            'height_success_timesteps': torch.zeros(self.num_envs),
            'height_success_count': 0,
            'obj_drop_count': 0,
            'episode_lengths': torch.zeros(self.num_envs),
            'episode_rewards': torch.zeros(self.num_envs),
            'total_timesteps': 0
        }

    def store_initial_positions(self, initial_obj_positions: torch.Tensor, newly_done):
        newly_done = newly_done.cpu()
        initial_obj_positions_clone = initial_obj_positions.clone().cpu()
        self.episode_data['initial_obj_positions'][newly_done] =initial_obj_positions_clone[newly_done]

    def get_position_std(self, positions: torch.Tensor) -> dict:
        return {
            'x_std': float(torch.std(positions[:, 0])),
            'y_std': float(torch.std(positions[:, 1])),
            'z_std': float(torch.std(positions[:, 2]))
        }

    def update_phase_data(self, not_visited_mask: torch.Tensor, timestep: int):
        for phase_idx in range(self.num_phases):
            newly_reached = ~not_visited_mask[:, phase_idx]
            for env_idx in range(self.num_envs):
                if newly_reached[env_idx]:
                    if self.episode_data['phase_reached_count'][phase_idx] == 0:
                        self.episode_data['phase_reached_count'][phase_idx] = 1
                        self.episode_data['phase_reached_timesteps'][phase_idx].append(timestep)

    def update_height_data(self, obj_positions: torch.Tensor, timestep: int):
        above_threshold = obj_positions[:, 2] > self.height_threshold
        #print("obj_positions", obj_positions)
        
        for env_idx in range(self.num_envs):
            if above_threshold[env_idx]:
                if self.episode_data['height_success_count'] == 0:
                    self.episode_data['height_success_timesteps'][env_idx] = timestep
                    self.episode_data['height_success_count'] = 1
                if self.episode_data['obj_drop_count'] == 1:
                    self.episode_data['obj_drop_count'] = 0
            
            else:
            
                if self.episode_data['height_success_count'] == 1 and timestep < 498:
                    self.episode_data['obj_drop_count'] = 1
    

    def finalize_episode(self, final_obj_positions: torch.Tensor, episode_lengths: torch.Tensor,
                         episode_rewards: torch.Tensor, total_timesteps: int):
        self.episode_data['final_obj_positions'] = final_obj_positions.clone()
        self.episode_data['episode_lengths'] = episode_lengths.clone()
        self.episode_data['episode_rewards'] = episode_rewards.clone()
        self.episode_data['total_timesteps'] = total_timesteps
        
        mask = self.episode_data['height_success_timesteps'] == 0
        self.episode_data['height_success_timesteps'][mask] = self.episode_data['episode_lengths'][mask].to(mask.device)
        self.all_episodes_data.append(self.episode_data.copy())

    def get_episode_summary(self, episode_idx: int) -> dict:
        if episode_idx >= len(self.all_episodes_data):
            return {}
        data = self.all_episodes_data[episode_idx]
        summary = {
            'episode': episode_idx,
            'total_timesteps': data['total_timesteps'],
            'phase_success_rates': {
                f'phase_{i+1}': float(data['phase_reached_count'][i]) 
                for i in range(self.num_phases)
            },
            'height_success_rate': float(data['height_success_count']),
            'avg_episode_length': float(torch.mean(data['episode_lengths'])),
            'avg_episode_reward': float(torch.mean(data['episode_rewards']))
           
        }
        return summary

    def get_overall_summary(self) -> dict:
        if not self.all_episodes_data:
            return {}
        num_episodes = len(self.all_episodes_data)
        phase_success_rates = {}
        for phase_idx in range(self.num_phases):
            phase_successes = torch.tensor([data['phase_reached_count'][phase_idx] for data in self.all_episodes_data], dtype=torch.float32)
            phase_success_rates[f'phase_{phase_idx+1}'] = {
                'mean': float(torch.mean(phase_successes)) / (len(self.all_episodes_data)) ,
                'std': float(torch.std(phase_successes)) / (len(self.all_episodes_data)) ,
                'rate': float(torch.sum(phase_successes)) / (len(self.all_episodes_data))
                
            }
        height_successes = torch.tensor([data['height_success_count'] for data in self.all_episodes_data], dtype=torch.float32)
        height_success_rate = {
            'mean': float(torch.mean(height_successes)) / (len(self.all_episodes_data)) ,
            'std': float(torch.std(height_successes)) / (len(self.all_episodes_data)) ,
            'rate': float(torch.sum(height_successes)) / (len(self.all_episodes_data)) 
        }

        drop_count = torch.tensor([data['obj_drop_count'] for data in self.all_episodes_data], dtype=torch.float32)
        drop_count_rate = {
            
            'rate': float(torch.sum(drop_count)) / (len(self.all_episodes_data)) 
        }
        end_timestaps = torch.cat([data['height_success_timesteps'] for data in self.all_episodes_data])
        episode_length_stats = {
            'mean': float(torch.mean(end_timestaps)),
            'std': float(torch.std(end_timestaps)),
            'min': float(torch.min(end_timestaps)),
            'max': float(torch.max(end_timestaps))
        }
        all_rewards = torch.cat([data['episode_rewards'] for data in self.all_episodes_data])
        reward_stats = {
            'mean': float(torch.mean(all_rewards)),
            'std': float(torch.std(all_rewards)),
            'min': float(torch.min(all_rewards)),
            'max': float(torch.max(all_rewards))
        }
        all_initial_positions = torch.cat([data['initial_obj_positions'] for data in self.all_episodes_data])
        all_final_positions = torch.cat([data['final_obj_positions'] for data in self.all_episodes_data])
        initial_position_std = self.get_position_std(all_initial_positions)
        final_position_std = self.get_position_std(all_final_positions)
        position_min = torch.min(all_initial_positions, dim=0).values
        position_max = torch.max(all_initial_positions, dim=0).values
        position_range = position_max - position_min
        summary = {
            'num_episodes': num_episodes,
            'num_envs_per_episode': self.num_envs,
            'total_trials': num_episodes * self.num_envs,
            'phase_success_rates': phase_success_rates,
            'height_success_rate': height_success_rate,
            'episode_length_stats': episode_length_stats,
            'reward_stats': reward_stats,
            'initial_position_std': initial_position_std,
            'final_position_std': final_position_std,
            'obj_drop_count_rate': drop_count_rate,
            'pos_min': position_min,
            'pos_max': position_max,
            'position_ranges': position_range
        }
        return summary


    def convert_tensors_to_python(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: self.convert_tensors_to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.convert_tensors_to_python(v) for v in obj]
        return obj
    

    def save_results(self, results_dir: str, task_name: str):
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{task_name}_{timestamp}"
        overall_summary = self.get_overall_summary()
        overall_summary = self.convert_tensors_to_python(overall_summary)

        with open(os.path.join(results_dir, f"{base_filename}_summary.json"), 'w') as f:
            json.dump(overall_summary, f, indent=2)

        episode_summaries = [self.get_episode_summary(i) for i in range(len(self.all_episodes_data))]
        with open(os.path.join(results_dir, f"{base_filename}_episodes.json"), 'w') as f:
            json.dump(episode_summaries, f, indent=2)
        serializable_data = []
        for episode_data in self.all_episodes_data:
            serializable_episode = {}
            for key, value in episode_data.items():
                if isinstance(value, torch.Tensor):
                    serializable_episode[key] = value.tolist()
                else:
                    serializable_episode[key] = value
            serializable_data.append(serializable_episode)
        with open(os.path.join(results_dir, f"{base_filename}_raw.json"), 'w') as f:
            json.dump(serializable_data, f, indent=2)
        print(f"Results saved to {results_dir}")
        print(f"Summary: {base_filename}_summary.json")
        print(f"Episodes: {base_filename}_episodes.json")
        print(f"Raw data: {base_filename}_raw.json")

def main():
    """Evaluate skrl agent with comprehensive metrics collection."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"], checkpoint="best_agent"
        )
    
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # configure and instantiate the skrl runner
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    # Initialize metrics collection
    metrics = EvaluationMetrics(
        num_envs=args_cli.num_envs,
        num_phases=5,  # Adjust based on your task
        height_threshold=args_cli.height_threshold,
        num_episodes=args_cli.num_episodes
    )

    print(f"[INFO] Starting evaluation with {args_cli.num_episodes} episodes, {args_cli.num_envs} environments each")
    print(f"[INFO] Height threshold: {args_cli.height_threshold}")
    robot = 'robot_1'
    # Main evaluation loop
    for episode in range(args_cli.num_episodes):
        print(f"\n[INFO] Starting episode {episode + 1}/{args_cli.num_episodes}")
        
        # Reset metrics for new episode
        metrics.reset_episode_data()
        
        # Reset environment and capture initial object positions
        obs, _ = env.reset()
        
        # Capture initial object positions right after reset
        
        
        timestep = 0
        episode_rewards = torch.zeros((args_cli.num_envs), dtype=torch.float, device=env.device)
        episode_lengths = torch.zeros((args_cli.num_envs), dtype=torch.float, device=env.device)
        dones = torch.zeros((args_cli.num_envs), dtype=torch.bool, device=env.device)
        
        # Episode loop
        final_obj_positions = torch.zeros((args_cli.num_envs, 3), dtype=torch.float, device=env.device) # Initialize storage for final positions
        
        while not torch.all(dones):
            start_time = time.time()
            
            # Run inference
            with torch.inference_mode():
                # Agent stepping
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                
                # Get actions
                if hasattr(env, "possible_agents"):
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])
                
                # Environment stepping
                obs, rewards, terminated, truncated, info = env.step(actions)
                   
                # Update episode data
                episode_rewards += rewards['robot_1'].squeeze(1) + rewards['robot_2'].squeeze(1)
            
                dones =  truncated['robot_1'] | truncated['robot_2']
                dones = dones.squeeze(1)
                # Update episode lengths for environments that just finished
                newly_done = dones & (episode_lengths == 0)
                episode_lengths[newly_done] = timestep + 1
                
                # Extract metrics from environment BEFORE any potential auto-reset
                current_obj_positions = None
                current_not_visited_mask = None
                

                initial_obj_positions = env.unwrapped.original_obj_positions.cpu()
                metrics.store_initial_positions(initial_obj_positions , newly_done)
                    
                current_not_visited_mask = env.unwrapped.not_visited_mask.cpu()
                metrics.update_phase_data(current_not_visited_mask, timestep)
            
                current_obj_positions = env.unwrapped._get_obj_pos().cpu()
                metrics.update_height_data(current_obj_positions, timestep)
            
                # Store final states for environments that just finished
                if torch.any(newly_done):
                    if current_obj_positions is not None:
                        # Store final object positions for newly finished environments
                        for env_idx in range(args_cli.num_envs):
                            if newly_done[env_idx]:
                                final_obj_positions[env_idx] = current_obj_positions[env_idx]
                    
                    # if current_not_visited_mask is not None:
                    #     # Store final phase completion status for newly finished environments
                    #     for env_idx in range(args_cli.num_envs):
                    #         if newly_done[env_idx]:
                    #             # Update final phase completion data
                    #             for phase_idx in range(6):  # Assuming 6 phases
                    #                 if not current_not_visited_mask[env_idx, phase_idx]:
                    #                     metrics.episode_data['phase_reached_count'][phase_idx] = 1
                
                timestep += 1
            
            # Time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
        
        # Finalize episode
        # Set remaining episode lengths for environments that didn't finish
        episode_lengths[episode_lengths == 0] = 9999
        
        # Get final object positions for any environments that didn't finish
        if hasattr(env.unwrapped, 'object_positions'):
            current_obj_positions = env.unwrapped.object_positions.cpu().numpy()
            # Fill in positions for environments that didn't finish naturally
            for env_idx in range(args_cli.num_envs):
                if episode_lengths[env_idx] == 9999:  # Didn't finish naturally
                    final_obj_positions[env_idx] = current_obj_positions[env_idx]
        
        metrics.finalize_episode(final_obj_positions, episode_lengths, episode_rewards, timestep)
        
        # Print episode summary
        episode_summary = metrics.get_episode_summary(episode)
        print(f"[INFO] Episode {episode + 1} completed:")
        print(f"  - Total timesteps: {episode_summary['total_timesteps']}")
        print(f"  - Average episode length: {episode_summary['avg_episode_length']:.2f}")
        print(f"  - Average reward: {episode_summary['avg_episode_reward']:.2f}")
        
        
        # Print position standard deviations
        # initial_std = episode_summary['initial_position_std']
        # final_std = episode_summary['final_position_std']
        # print(f"  - Initial position std: X={initial_std['x_std']:.4f}, Y={initial_std['y_std']:.4f}, Z={initial_std['z_std']:.4f}")
        # print(f"  - Final position std: X={final_std['x_std']:.4f}, Y={final_std['y_std']:.4f}, Z={final_std['z_std']:.4f}")
        
        for phase, rate in episode_summary['phase_success_rates'].items():
            print(f"  - {phase} success rate: {rate:.2}")
        print(f"  - Height success rate: {episode_summary['height_success_rate']:.2}")
    # Save results
    metrics.save_results(args_cli.results_dir, args_cli.task)
    
    # Print overall summary
    overall_summary = metrics.get_overall_summary()
    print(f"\n[INFO] Overall Evaluation Results:")
    print(f"  - Total trials: {overall_summary['total_trials']}")
    print(f"  - Episodes: {overall_summary['num_episodes']}")
    print(f"  - Environments per episode: {overall_summary['num_envs_per_episode']}")
    
    print(f"\n[INFO] Position Deviations:")
    print(f"  -Displacement: X={overall_summary['position_ranges'][0]:.4f}, "
          f"Y={overall_summary['position_ranges'][1]:.4f}, Z={overall_summary['position_ranges'][2]:.4f}")
    
    print(f"\n[INFO] Phase Success Rates (mean ± std):")
    for phase, stats in overall_summary['phase_success_rates'].items():
        print(f"  - {phase}: {stats['rate']:.2}")
    print(f"\n[INFO] Height Success Rate: {overall_summary['height_success_rate']['rate']:.2}")
    print(f"\n[INFO] Obj Drop Rate: {overall_summary['obj_drop_count_rate']['rate']:.2}")
    print(f"\n[INFO] Average Episode Length: {overall_summary['episode_length_stats']['mean']:.2f} ± {overall_summary['episode_length_stats']['std']:.2f}")
    print(f"[INFO] Average Reward: {overall_summary['reward_stats']['mean']:.2f} ± {overall_summary['reward_stats']['std']:.2f}")

    # Close environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()