# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
ShadowHand Over environment.
"""

import gymnasium as gym

from . import agents
from . import handover_env, handover_env_cfg, joint_pos_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Handover-Block-Dual-PSM-Direct-v0",
    entry_point=handover_env.DualArmHandoverEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.BlockHandoverEnvCfg,
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Handover-Block-Dual-PSM-Play-Direct-v0",
    entry_point=handover_env.DualArmHandoverEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.BlockHandoverEnvCfg_PLAY,
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)