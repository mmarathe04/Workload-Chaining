# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import numpy as np
import torch

import cv2

import matplotlib
matplotlib.use('Agg')
#import torchvision.transforms as transforms
from types import SimpleNamespace
from viplanner.config.learning_cfg import TrainCfg
from viplanner.plannernet import AutoEncoder, DualAutoEncoder, get_m2f_cfg
from viplanner.traj_cost_opt.traj_opt import TrajOpt

torch.set_default_dtype(torch.float32)

cfg = SimpleNamespace()
cfg.model_save = "/scratch/mihir/viplanner/ros/planner/src"
cfg.m2f_config_path = "/scratch/mihir/viplanner/ros/planner/src/model.yaml" #added this to fix line 48 error for config path
#cfg.m2f_config_path = "model.yaml"

class VIPlannerInference:
    def __init__(
        self,
        cfg,
    ) -> None:
        """VIPlanner Inference Script

        Args:
            cfg (Namespace): Config Namespace
        """
        # get configs
        model_path = os.path.join(cfg.model_save, "model.pt")
        config_path = os.path.join(cfg.model_save, "model.yaml")

        # get train config
        #self.train_cfg: TrainCfg = TrainCfg.from_yaml(cfg.m2f_config_path)
        #self.train_cfg: TrainCfg = TrainCfg(**cfg.m2f_config_path["config"])
        if isinstance(cfg.m2f_config_path, str):
            # It's a path to a YAML file
            self.train_cfg: TrainCfg = TrainCfg.from_yaml(cfg.m2f_config_path)
        elif isinstance(cfg.m2f_config_path, dict):
            # It's already a parsed dictionary
            self.train_cfg: TrainCfg = TrainCfg(**cfg.m2f_config_path["config"])
        else:
            raise TypeError(f"Unexpected type for m2f_config_path: {type(cfg.m2f_config_path)}")

        print(f"TrainConfig - RGB: {self.train_cfg.rgb}, SEM: {self.train_cfg.sem}")
        print(f"Using DualAutoEncoder: {self.train_cfg.rgb or self.train_cfg.sem}")


        # get model
        if self.train_cfg.rgb: #or self.train_cfg.sem:  #the or self.train_cfg.sem is a debug statement
            print("Initializing DualAutoEncoder...")
            m2f_cfg = get_m2f_cfg(cfg.m2f_config_path)
            self.pixel_mean = m2f_cfg.MODEL.PIXEL_MEAN
            self.pixel_std = m2f_cfg.MODEL.PIXEL_STD
        else:
            print("Initializing AutoEncoder...")
            m2f_cfg = None
            self.pixel_mean = [0, 0, 0]
            self.pixel_std = [1, 1, 1]

        if self.train_cfg.rgb or self.train_cfg.sem:
            self.net = DualAutoEncoder(train_cfg=self.train_cfg, m2f_cfg=m2f_cfg)
        else:
            self.net = AutoEncoder(
                encoder_channel=self.train_cfg.in_channel,
                k=self.train_cfg.knodes,
            )
        try:
            model_state_dict, _ = torch.load(model_path, map_location=torch.device('cpu'))
        except ValueError:
            model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.net.load_state_dict(model_state_dict)

        # inference script = no grad for model
        self.net.eval()

        # move to GPU if available
        if torch.cuda.is_available():
            self.net = self.net.cuda()
            self._device = "cuda"
        else:
            self._device = "cpu"

        # transforms
        #self.transforms = transforms.Compose(
         #   [
          #      transforms.ToTensor(),
           #     transforms.Resize(tuple(self.train_cfg.img_input_size)),
            #]
        #)

        # get trajectory generator
        self.traj_generate = TrajOpt()
        return

    # def img_converter(self, img: np.ndarray) -> torch.Tensor:
    #     # crop image and convert to tensor
    #     img = self.transforms(img)
    #     return img.unsqueeze(0).to(self._device)

    def plan(
        self,
        depth_image: np.ndarray,
        sem_rgb_image: np.ndarray,
        goal_robot_frame: torch.Tensor,
    ) -> tuple:
        """Plan to path towards the goal given depth and semantic image

        Args:
            depth_image (np.ndarray): Depth image from the robot
            goal_robot_frame (torch.Tensor): Goal in robot frame
            sem_rgb_image (np.ndarray): Semantic/ RGB Image from the robot.

        Returns:
            tuple: _description_
        """

        #print("Testing forward pass with random inputs for shape validation...")

        # print(f"Processed depth_image shape: {depth_image.shape}")  
        # print(f"Processed sem_rgb_image shape: {sem_rgb_image.shape}")

        # print(f"Goal Tensor Shape: {goal_robot_frame.shape}")

        # this was originally here
        # depth_image = torch.rand(2, 1, 640, 384).to(self._device)  
        # #sem_rgb_image = torch.rand(2, 3, 640, 384).to(self._device)  
        # sem_rgb_image = torch.rand(2, 3, 640, 384).to(self._device)
        # #goal_robot_frame = torch.rand(2, 3).to(self._device)
        # goal_robot_frame = torch.tensor([
        #     [1.0, 0.0, 0.0],  # Goal for sample 1
        #     [1.0, 0.0, 0.0],  # Goal for sample 2
        # ], dtype=torch.float32).to(self._device)

        # depth_image = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED)
        # sem_rgb_image = cv2.imread("segmentation.png", cv2.IMREAD_UNCHANGED)

        # depth_tensor = torch.tensor(depth_image, dtype=torch.float32).unsqueeze(0)  

        # segmentation_tensor = torch.tensor(sem_rgb_image, dtype=torch.float32).unsqueeze(0)

        # model = VIPModel()

        # output = model(depth_tensor, segmentation_tensor)
        # print("Model Output:", output)

        # depth_image = torch.reand(2, 1, 480, 640).to(self._device)  
        # sem_rgb_image = torch.rand(2, 3, 480, 640).to(self._device)  
        # goal_robot_frame = torch.rand(2, 3).to(self._device)

        # depth_image = torch.rand(2, 1, 640, 384).to(self._device)  
        # sem_rgb_image = torch.rand(1, 3, 640, 384).to(self._device)  
        # goal_robot_frame = torch.rand(1, 3).to(self._device)

        # depth_image = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0).float().to(self._device)
        # sem_rgb_image = torch.from_numpy(sem_rgb_image).permute(2, 0, 1).unsqueeze(0).float().to(self._device)

        # sem_rgb_image = sem_rgb_image.permute(0, 3, 1, 2)
        # depth_image = depth_image.permute(0, 3, 1, 2)

        # np_image = sem_rgb_image.detach().cpu().numpy()
        # np_image = np.transpose(np_image[0], (1, 2, 0))

        depth_image = torch.from_numpy(depth_image).permute(2, 0, 1).unsqueeze(0).float().to(self._device)  # [1, C, H, W]
        sem_rgb_image = torch.from_numpy(sem_rgb_image).permute(2, 0, 1).unsqueeze(0).float().to(self._device)  # [1, C, H, W]

        with torch.no_grad():
            #print("Calling forward pass in the AutoEncoder model...")

            keypoints, fear = self.net(depth_image, sem_rgb_image, goal_robot_frame.to(self._device))
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            keypoints_np = keypoints.cpu().numpy()

            goal_np = goal_robot_frame.detach().cpu().numpy()

            for i in range(keypoints_np.shape[0]):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                # Plot keypoints
                ax.scatter(keypoints_np[i, :, 0], keypoints_np[i, :, 1], keypoints_np[i, :, 2],
                        label='Keypoints', c='blue', s=50)

                # Plot goal
                ax.scatter(goal_np[i, 0], goal_np[i, 1], goal_np[i, 2],
                        label='Goal', c='red', marker='*', s=100)

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f'3D Keypoints and Goal - Sample {i+1}')
                ax.legend()
                plt.savefig(f"keypoints_sample_{i+1}.png")
                plt.close()
            print("Keypoints shape:", keypoints.shape)
            print("Keypoints:", keypoints)

        print("Fear:", fear)
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)
        print("Traj:", traj)

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Your trajectory tensor (on CUDA) — move to CPU and squeeze
        traj_np = traj.squeeze(0).cpu().numpy()  # shape: (N, 3)

        # Extract x, y, z
        x = traj_np[:, 0]
        y = traj_np[:, 1]
        z = traj_np[:, 2]

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, marker='o')

        # Optional: set labels and view angle
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=30, azim=135)  # adjust viewing angle

        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()

        plt.savefig("Trajectory.png")
        plt.close()

        return traj.cpu().squeeze(0).numpy(), fear.cpu().numpy()

        

    def plan_depth(
        self,
        depth_image: np.ndarray,
        goal_robot_frame: torch.Tensor,
    ) -> tuple:

        # print(f"Depth Image Shape before net (plan_depth): {depth_image.shape}")
        # print(f"Goal Tensor Shape (plan_depth): {goal_robot_frame.shape}")

        with torch.no_grad():
            #depth_image = self.img_converter(depth_image).float()
            keypoints, fear = self.net(depth_image, goal_robot_frame.to(self._device))

        # generate trajectory
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)

        return traj.cpu().squeeze(0).numpy(), fear.cpu().numpy()

        

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"device: {device}")

    # image_height = 480 
    # image_width = 640 
    # depth_image = np.random.rand(image_height, image_width).astype(np.float32) 
    # sem_rgb_image = np.random.randint(0, 256, (image_height, image_width, 3), dtype=np.uint8) 

    image_height = 384 
    image_width = 640 
    depth_image = np.random.rand(image_height, image_width).astype(np.float32) 
    sem_rgb_image = np.random.randint(0, 256, (image_height, image_width, 3), dtype=np.uint8)

    goal_robot_frame = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    ViPlanner = VIPlannerInference(cfg)
    plan1 = ViPlanner.plan(depth_image, sem_rgb_image, goal_robot_frame)


    #adding this
    
# EoF
