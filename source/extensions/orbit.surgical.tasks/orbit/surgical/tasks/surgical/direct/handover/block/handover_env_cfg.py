from isaaclab.utils import configclass
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import mdp
from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg

from dataclasses import MISSING


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the handover scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot_1: ArticulationCfg = MISSING
    robot_2: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_1_frame: FrameTransformerCfg = MISSING
    ee_2_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.457)),
        spawn=UsdFileCfg(usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Table/table.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.95)),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.1, -0.1)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    body_1_joint_pos: mdp.JointPositionActionCfg = MISSING
    finger_1_joint_pos: mdp.BinaryJointPositionActionCfg = MISSING
    body_2_joint_pos: mdp.JointPositionActionCfg = MISSING
    finger_2_joint_pos: mdp.BinaryJointPositionActionCfg = MISSING

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_1_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot_1",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.05, 0.05),
            pos_y=(-0.05, 0.05),
            pos_z=(-0.12, -0.08),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )
    # set the scale of the visualization markers to (0.01, 0.01, 0.01)
    ee_1_pose.goal_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
    ee_1_pose.current_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)

    ee_2_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot_2",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.05, 0.05),
            pos_y=(-0.05, 0.05),
            pos_z=(-0.12, -0.08),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )
    # set the scale of the visualization markers to (0.01, 0.01, 0.01)
    ee_2_pose.goal_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
    ee_2_pose.current_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)



@configclass
class DualArmHandoverEnvCfg(DirectMARLEnvCfg):
    # Agents
    possible_agents = ["robot_1", "robot_2"]
    # action - 7, obs - 33 in the manager
    action_spaces = {"robot_1": 8, "robot_2": 8}  # IK target delta pose
    observation_spaces = {"robot_1": 33, "robot_2": 33}  # example dim (can be tuned)
    state_space = 66  # combined
    ground_height=0.0149
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    body_joint_names = (
        "psm_yaw_joint",
        "psm_pitch_end_joint",
        "psm_main_insertion_joint",
        "psm_tool_roll_joint",
        "psm_tool_pitch_joint",
        "psm_tool_yaw_joint",
    )

    finger_joint_names = (
        "psm_tool_gripper1_joint",
        "psm_tool_gripper2_joint",
    )
    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        render_interval=2,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )

    # Robots


    # Object to hand over
   
    # Scene
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)

    # Constants for logic (used in reward, reset, etc.)
    ee_link_name: str = "psm_tool_tip_link"
    reset_position_noise = 0.01
    reset_rot_noise = 0.1
    fall_z_threshold = -0.05
    dist_reward_scale = 20.0
    act_moving_average = 1.0
    episode_length_s = 15

    actions: ActionsCfg = ActionsCfg()
    # Action/observation scaling constants
    ik_scale = 0.05

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 15.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.viewer.eye = (0.0, 0.5, 0.2)
        self.viewer.lookat = (0.0, 0.0, 0.05)

