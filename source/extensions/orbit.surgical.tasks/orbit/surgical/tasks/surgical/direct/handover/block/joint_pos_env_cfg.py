from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import PhysicsMaterialCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.envs import mdp


from .handover_env_cfg import DualArmHandoverEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from orbit.surgical.assets.psm import PSM_CFG  # isort: skip


@configclass
class BlockHandoverEnvCfg(DualArmHandoverEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Set PSM as robot
        self.scene.robot_1 = PSM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
        self.scene.robot_1.init_state.pos = (0.18, 0.0, 0.15)
        #self.scene.robot_1.init_state.pos = (-0.0076, -0.0055,  0.1)
        self.scene.robot_1.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot_2 = PSM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
        self.scene.robot_2.init_state.pos = (-0.18, 0.0, 0.15)
        self.scene.robot_2.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        # Set actions for the specific robot type (PSM)
        self.actions.body_1_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot_1",
            joint_names=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.body_2_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot_2",
            joint_names=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.finger_1_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_1",
            joint_names=["psm_tool_gripper.*_joint"],
            open_command_expr={"psm_tool_gripper1_joint": -0.5, "psm_tool_gripper2_joint": 0.5},
            close_command_expr={"psm_tool_gripper1_joint": -0.07, "psm_tool_gripper2_joint": 0.07},
        )
        self.actions.finger_2_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_2",
            joint_names=["psm_tool_gripper.*_joint"],
            open_command_expr={"psm_tool_gripper1_joint": -0.5, "psm_tool_gripper2_joint": 0.5},
            close_command_expr={"psm_tool_gripper1_joint": -0.07, "psm_tool_gripper2_joint": 0.07},
        )
        # Set the body name for the end effector
        self.commands.ee_1_pose.body_name = "psm_tool_tip_link"
        self.commands.ee_2_pose.body_name = "psm_tool_tip_link"

        # Set Peg Block as object
        self.scene.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.2, 0.0, 0.05), rot=(1, 0, 0, 0)),
        spawn=UsdFileCfg(
            usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Surgical_block/block_backup.usda",
            scale=(0.011, 0.011, 0.011),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=64,  # Higher for small objects
                solver_velocity_iteration_count=32,
                max_angular_velocity=0.1,  # Allow natural movement
                max_linear_velocity=0.1,   # Allow natural movement
                max_depenetration_velocity=0.09,  # Higher to resolve penetration faster
                disable_gravity=False,
                linear_damping=0.2,  # Light damping to prevent bouncing
                angular_damping=0.3,  # Light rotational damping
            ),
            # Add physics material for the object
            
            # Add collision properties
            collision_props=CollisionPropertiesCfg(
                contact_offset=0.0015,  # Small but sufficient for tiny object
                rest_offset=0.001,     # Very small rest offset
            ),
        ),
    )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_1_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_1/psm_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_1/psm_tool_tip_link",
                    name="end_effector",
                ),
            ],
        )
        self.scene.ee_2_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_2/psm_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_2/psm_tool_tip_link",
                    name="end_effector",
                ),
            ],
        )
        self.is_training = True
        
    


@configclass
class BlockHandoverEnvCfg_PLAY(BlockHandoverEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.is_training = False
        self.episode_length_s = 15
        # disable randomization for play
        #self.observations.policy.enable_corruption = False
