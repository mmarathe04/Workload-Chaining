# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.utils import configclass
from omni.viplanner.utils import UnRealImporterCfg

from ..viplanner import DATA_DIR
from .base_cfg import ViPlannerBaseCfg

##
# Pre-defined configs
##
# isort: off
from omni.isaac.lab_assets.anymal import ANYMAL_C_CFG


##
# Scene definition
##


@configclass
class TerrainSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = UnRealImporterCfg(
        prim_path="/World/Warehouse",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        usd_path=os.path.join(DATA_DIR, "warehouse", "warehouse_new.usd"),
        groundplane=True,
        sem_mesh_to_class_map=os.path.join(DATA_DIR, "warehouse", "keyword_mapping.yml"),
        people_config_file=os.path.join(DATA_DIR, "warehouse", "people_cfg.yml"),
        axis_up="Z",
    )

    # robots
    robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (5.0, 5.5, 0.6)
    robot.init_state.rot = (0.5253, 0.0, 0.0, 0.8509)

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/GroundPlane"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, debug_vis=False)
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=1000.0,
        ),
    )
    # camera
    depth_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/depth_camera",
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(-0.5, 0.5, -0.5, 0.5)),
        spawn=sim_utils.PinholeCameraCfg(),
        width=848,
        height=480,
        data_types=["distance_to_image_plane"],
    )

    # NOTE: remove "rgb" from the data_types to only render the semantic segmentation
    semantic_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/semantic_camera",
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(-0.5, 0.5, -0.5, 0.5)),
        spawn=sim_utils.PinholeCameraCfg(),
        width=1280,
        height=720,
        data_types=["semantic_segmentation", "rgb"],
        colorize_semantic_segmentation=False,
    )


##
# Environment configuration
##


@configclass
class ViPlannerWarehouseCfg(ViPlannerBaseCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs=1, env_spacing=1.0)

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # adapt viewer
        self.viewer.eye = (5, 12, 5)
        self.viewer.lookat = (5, 0, 0.0)

        #added this#
        #self.assign_numeric_semantics()
        ###

    def assign_numeric_semantics(self):
        """Assign numeric semantic labels based on keyword mapping."""
        import yaml
        import omni.usd
        from pxr import UsdGeom
        import omni.kit.commands
        import os

        print("[DEBUG] Assigning semantic classes from keyword_mapping.yml")
        print(f"[DEBUG] Classes found: {self.scene.terrain.sem_mesh_to_class_map}")


        # Load mapping
        mapping_file = os.path.join(DATA_DIR, "warehouse", "keyword_mapping.yml")
        with open(mapping_file, "r") as f:
            keyword_mapping = yaml.safe_load(f)

        label_to_id = {label: idx+1 for idx, label in enumerate(keyword_mapping.keys())}
        mesh_to_id = {}
        for label, prefixes in keyword_mapping.items():
            for prefix in prefixes:
                mesh_to_id[prefix] = label_to_id[label]

        def recursive_assign_numeric_semantics(prim, mesh_to_id):
            if prim.IsA(UsdGeom.Xform) or prim.IsA(UsdGeom.Mesh):
                prim_name = prim.GetName()
                assigned_id = None
                for prefix, sem_id in mesh_to_id.items():
                    if prim_name.startswith(prefix):
                        assigned_id = sem_id
                        break
                if assigned_id is not None:
                    omni.kit.commands.execute(
                        "AddSemanticLabel",
                        path=prim.GetPath().pathString,
                        semanticLabel=str(assigned_id),
                    )
            for child in prim.GetChildren():
                recursive_assign_numeric_semantics(child, mesh_to_id)

        # Apply
        stage = omni.usd.get_context().get_stage()
        warehouse_prim = stage.GetPrimAtPath("/World/Warehouse")
        recursive_assign_numeric_semantics(warehouse_prim, mesh_to_id)
