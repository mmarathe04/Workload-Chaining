# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to add and simulate on-board sensors for a robot.

We add the following sensors on the quadruped robot, ANYmal-C (ANYbotics):

* USD-Camera: This is a camera sensor that is attached to the robot's base.
* Height Scanner: This is a height scanner sensor that is attached to the robot's base.
* Contact Sensor: This is a contact sensor that is attached to the robot's feet.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/04_sensors/add_sensors_on_robot.py --enable_cameras

"""

"""Launch Isaac Sim Simulator first."""

import sys
sys.path.append("/scratch/mihir/IsaacLab/source/viplanner/ros/planner/src")

#from vip_inference_copy import VIPlannerInference

import cv2
import numpy as np

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on adding sensors on a robot.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.utils import configclass

##
# Pre-defined configs
##
from omni.isaac.lab_assets.anymal import ANYMAL_C_CFG  # isort: skip


@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        update_period=0.1,
        height = 384,
        width = 640,
        # height=480,
        # width=640,
        #semantic_segmentation=True,
        data_types=["semantic_segmentation", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/defaultGroundPlane"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_FOOT", update_period=0.0, history_length=6, debug_vis=True
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""

    from types import SimpleNamespace
    from vip_inference_copy import VIPlannerInference

    cfg = SimpleNamespace()
    cfg.model_save = "/scratch/mihir/viplanner/ros/planner/src"
    cfg.m2f_config_path = "/scratch/mihir/viplanner/ros/planner/src/model.yaml"

    viplanner_inst = VIPlannerInference(cfg)


    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Simulate physics
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_state_to_sim(root_state)
            # set joint positions with some noise
            joint_pos, joint_vel = (
                scene["robot"].data.default_joint_pos.clone(),
                scene["robot"].data.default_joint_vel.clone(),
            )
            joint_pos += torch.rand_like(joint_pos) * 0.1
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply default actions to the robot
        # -- generate actions/commands
        targets = scene["robot"].data.default_joint_pos
        # -- apply action to the robot
        scene["robot"].set_joint_position_target(targets)
        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

        # print information from the sensors
        print("-------------------------------")
        print(scene["camera"])
        print("Received shape of segmentation   image: ", scene["camera"].data.output["semantic_segmentation"].shape)
        print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)
        print("-------------------------------")
        print(scene["height_scanner"])
        print("Received max height value: ", torch.max(scene["height_scanner"].data.ray_hits_w[..., -1]).item())
        print("-------------------------------")
        print(scene["contact_forces"])
        print("Received max contact force of: ", torch.max(scene["contact_forces"].data.net_forces_w).item())

        
        depth_tensor = scene["camera"].data.output["distance_to_image_plane"]
        segmentation_tensor = scene["camera"].data.output["semantic_segmentation"]

        max_val = 100000
        min_val = 0.1
        depth_tensor = torch.nan_to_num(depth_tensor, nan=0.0, posinf=max_val)
        depth_tensor = torch.clamp(depth_tensor, min_val, max_val)
        #depth_tensor = (depth_tensor - min_val) / (max_val - min_val)

        if segmentation_tensor.shape[-1] == 4:
            segmentation_tensor = segmentation_tensor[..., :3]

        goal_tensor = torch.tensor([1.0, 1.0, 0.0])
        goal_tensor = goal_tensor.unsqueeze(0)

        print("segmentation image:", segmentation_tensor)
        print("depth image:", depth_tensor)

        depth_image = depth_tensor.squeeze(0).detach().cpu().numpy()  
        segmentation_image = segmentation_tensor.squeeze(0).detach().cpu().numpy() 

        cv2.imwrite("segmentation.png", segmentation_image)
        cv2.imwrite("depth_vis.png", depth_image)

        plan2 = viplanner_inst.plan(depth_image, segmentation_image, goal_tensor)


        # depth_normalized = depth_image - depth_image.min()
        # depth_normalized = depth_normalized / (depth_normalized.max() + 1e-8)  # avoid divide-by-zero
        # depth_uint8 = (depth_normalized * 255).astype(np.uint8)

        # cv2.imwrite(f"depth_{count}.png", (depth_image * 255).astype(np.uint8))
        # cv2.imwrite(f"segmentation_{count}.png", segmentation_image)

        #segmentation_image = np.nan_to_num(segmentation_image, nan=0.0, posinf=0.0)
        #segmentation_image = segmentation_image / 255.0
        #segmentation_image = np.clip(segmentation_image, 0, 255)

        # segmentation_image = segmentation_image[..., :3]
        # segmentation_image = segmentation_image.float() / 255.0


        # depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=5.0)
        # valid_depth = np.clip(depth_image, 0.1, 5.0)        
        # normalized_depth = (valid_depth - 0.1) / (5.0 - 0.1)


        # cv2.imwrite(f"depth_{count}.png", (depth_image * 255).astype(np.uint8))
        # cv2.imwrite(f"segmentation_{count}.png", segmentation_image)

        # depth_image = cv2.imread(f"depth_{count}.png", cv2.IMREAD_UNCHANGED)
        # segmentation_image = cv2.imread(f"segmentation_{count}.png", cv2.IMREAD_UNCHANGED)



        # print(f"[INFO] Saved images for step {count}")
        # cv2.imwrite(f"depth_{count:04d}.png", (depth_image * 255).astype(np.uint8))
        # cv2.imwrite(f"segmentation_{count:04d}.png", segmentation_image)

        # print(f"Segmentation tensor shape at step {count}: {segmentation_tensor.shape}")
        # print(f"Depth tensor shape at step {count}: {depth_tensor.shape}"

        
        # depth_image = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED)
        # segmentation_image = cv2.imread("segmentation.png", cv2.IMREAD_UNCHANGED)

        # print(f"Depth image type: {type(depth_image)}, shape: {getattr(depth_image, 'shape', None)}")
        # print(f"Segmentation image type: {type(segmentation_image)}, shape: {getattr(segmentation_image, 'shape', None)}")

        # plan2 = viplanner_inst.plan(depth_image, segmentation_image, goal_tensor)


        

        # depth_image = depth_tensor.detach().cpu().numpy()
        # segmentation_image = segmentation_tensor.detach().cpu().numpy()

        # # cv2.imwrite("depth.png", (depth_image * 255).astype(np.uint8))
        # # cv2.imwrite("segmentation.png", segmentation_image)

        # cv2.imwrite(f"depth_{count:04d}.png", (depth_image * 255).astype(np.uint8))
        # cv2.imwrite(f"segmentation_{count:04d}.png", segmentation_image)
        # print(f"[INFO] Saved images for step {count}")

        # if (
        #     depth_tensor is None or segmentation_tensor is None or
        #     depth_tensor.numel() == 0 or segmentation_tensor.numel() == 0
        # ):
        #     print(f"[WARNING] Empty image data at step {count}, skipping save.")
        #     continue

        # # Convert to numpy
        # depth_image = depth_tensor.detach().cpu().numpy()
        # segmentation_image = segmentation_tensor.detach().cpu().numpy()

        # Handle shape errors
        # if len(depth_image.shape) != 1 or len(segmentation_image.shape) != 4:
        #     print(f"[WARNING] Invalid image shape at step {count}, skipping save.")
        #     continue

        # Scale and save
        # cv2.imwrite(f"depth_{count:04d}.png", (depth_image * 255).astype(np.uint8))
        # cv2.imwrite(f"segmentation_{count:04d}.png", segmentation_image)
        # print(f"[INFO] Saved images for step {count}")


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = SensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
