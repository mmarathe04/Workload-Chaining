# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the rigid objects class.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

import sys
sys.path.append("/scratch/mihir/IsaacLab/source/viplanner/ros/planner/src")

import cv2
import numpy as np

# omni-isaac-lab
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument("--conv_distance", default=0.2, type=float, help="Distance for a goal considered to be reached.")
parser.add_argument(
    "--scene", default="warehouse", choices=["matterport", "carla", "warehouse"], type=str, help="Scene to load."
)
parser.add_argument("--model_dir", default=None, type=str, help="Path to model directory.")

# add applauncher arguments
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import omni.isaac.core.utils.prims as prim_utils
import torch
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.viplanner.config import (
    ViPlannerCarlaCfg,
    ViPlannerMatterportCfg,
    ViPlannerWarehouseCfg,
)
from omni.viplanner.viplanner import VIPlannerAlgo
from pxr import UsdGeom

"""
Main
"""
##
from pxr import Usd

usd_path = "/scratch/mihir/IsaacLab/source/viplanner/omniverse/extension/omni.viplanner/data/warehouse/warehouse_new.usd"

stage = Usd.Stage.Open(usd_path)

stage.Export(usd_path.replace(".usd", ".usda"))

print("Exported .usda successfully!")
##

def main():
    """Imports all legged robots supported in IsaacLab and applies zero actions."""

    # create environment cfg
    if args_cli.scene == "matterport":
        env_cfg = ViPlannerMatterportCfg(seed=1234)
        goal_pos = torch.tensor([8.0, -13.5, 1.0])
    elif args_cli.scene == "carla":
        env_cfg = ViPlannerCarlaCfg(seed=1234)
        goal_pos = torch.tensor([137, 111.0, 1.0])
    elif args_cli.scene == "warehouse":
        env_cfg = ViPlannerWarehouseCfg(seed=1234)
        goal_pos = torch.tensor([3, -4.5, 1.0])
    else:
        raise NotImplementedError(f"Scene {args_cli.scene} not yet supported!")

    # create environment
    env = ManagerBasedRLEnv(env_cfg)

    # adjust the intrinsics of the camera
    depth_intrinsic = torch.tensor([[430.31607, 0.0, 428.28408], [0.0, 430.31607, 244.00695], [0.0, 0.0, 1.0]])
    env.scene.sensors["depth_camera"].set_intrinsic_matrices(matrices=depth_intrinsic.repeat(env.num_envs, 1, 1))
    semantic_intrinsic = torch.tensor([[644.15496, 0.0, 639.53125], [0.0, 643.49212, 366.30880], [0.0, 0.0, 1.0]])
    env.scene.sensors["semantic_camera"].set_intrinsic_matrices(matrices=semantic_intrinsic.repeat(env.num_envs, 1, 1))

    # Make sure that groundplane is invisible
    if args_cli.scene == "carla":
        assert (
            prim_utils.get_prim_at_path("/World/GroundPlane").GetAttribute("visibility").Set(UsdGeom.Tokens.invisible)
        )

    # reset the environment
    with torch.inference_mode():
        obs = env.reset()[0]

    #added this
    if args_cli.scene == "warehouse":
        env_cfg.assign_numeric_semantics()
    ###


    # set goal cube
    VisualCuboid(
        prim_path="/World/goal",  # The prim path of the cube in the USD stage
        name="waypoint",  # The unique name used to retrieve the object from the scene later on
        position=goal_pos,  # Using the current stage units which is in meters by default.
        scale=torch.tensor([0.15, 0.15, 0.15]),  # most arguments accept mainly numpy arrays.
        size=1.0,
        color=torch.tensor([1, 0, 0]),  # RGB channels, going from 0-1
    )
    goal_pos = prim_utils.get_prim_at_path("/World/goal").GetAttribute("xformOp:translate")

    # pause the simulator
    # env.sim.pause()

    # load viplanner
    viplanner = VIPlannerAlgo(model_dir=args_cli.model_dir, device=env.device)

    goals = torch.tensor(goal_pos.Get(), device=env.device).repeat(env.num_envs, 1)

    # initial paths
    _, paths, fear = viplanner.plan_dual(
        obs["planner_image"]["depth_measurement"], obs["planner_image"]["semantic_measurement"], goals
    )

    # Simulate physics
    while simulation_app.is_running():

        # from vip_inference_copy import VIPlannerInference

        # cfg = SimpleNamespace()
        # cfg.model_save = "/scratch/mihir/viplanner/ros/planner/src"
        # cfg.m2f_config_path = "/scratch/mihir/viplanner/ros/planner/src/model.yaml"

        # viplanner_inst = VIPlannerInference(cfg)

        with torch.inference_mode():
            # If simulation is paused, then skip.
            if not env.sim.is_playing():
                env.sim.step(render=~args_cli.headless)
                continue

            obs = env.step(action=paths.view(paths.shape[0], -1))[0]

        # apply planner
        goals = torch.tensor(goal_pos.Get(), device=env.device).repeat(env.num_envs, 1)
        if torch.any(
            torch.norm(obs["planner_transform"]["cam_position"] - goals)
            > viplanner.train_config.data_cfg[0].max_goal_distance
        ):
            print(
                f"[WARNING]: Max goal distance is {viplanner.train_config.data_cfg[0].max_goal_distance} but goal is {torch.norm(obs['planner_transform']['cam_position'] - goals)} away from camera position! Please select new goal!"
            )
            env.sim.pause()
            continue

        goal_cam_frame = viplanner.goal_transformer(
            goals, obs["planner_transform"]["cam_position"], obs["planner_transform"]["cam_orientation"]
        )
        _, paths, fear = viplanner.plan_dual(
            obs["planner_image"]["depth_measurement"], obs["planner_image"]["semantic_measurement"], goal_cam_frame
        )
        paths = viplanner.path_transformer(
            paths, obs["planner_transform"]["cam_position"], obs["planner_transform"]["cam_orientation"]
        )

        # draw path
        viplanner.debug_draw(paths, fear, goals)

        depth_tensor = obs["planner_image"]["depth_measurement"]
        segmentation_tensor = obs["planner_image"]["semantic_measurement"]

        print("segmentation image:", segmentation_tensor)
        print("depth image:", depth_tensor)
        print(obs["planner_image"].keys())

        if segmentation_tensor.shape[-1] == 4:
            segmentation_tensor = segmentation_tensor[..., :3]

        depth_image = depth_tensor.squeeze(0).detach().cpu().numpy()  
        segmentation_image = segmentation_tensor.squeeze(0).detach().cpu().numpy() 

        cv2.imwrite("segmentation_img.png", segmentation_image)
        cv2.imwrite("depth_img.png", depth_image)

if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
