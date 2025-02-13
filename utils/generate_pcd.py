import os
import cv2
import json
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from torchvision.transforms import ToTensor

from utils.readCam import readCamerasFromTransforms
from utils.graphics_utils import getWorld2View2

def get_pointcloud(color, depth, w2c, FX, FY, CX, CY, transform_pts=True, mask=None):
    width, height = color.shape[2], color.shape[1]
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth.reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        c2w = torch.inverse(w2c)
        R = c2w[:3, :3]
        T = c2w[:3, 3]
        pts = (R @ pts_cam.T) + T.unsqueeze(1)
        pts = pts.T
    else:
        pts = pts_cam
    
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)# (C, H, W) -> (H, W, C)
    point_cld = torch.cat((pts, cols), -1)

    # # Select points based on mask
    mask1 = (depth_z > 0) & (depth_z < 0.5)
    mask2 = (depth_z > 0.5) & (depth_z < 5)
    fore_pcd = point_cld[mask1]
    back_pcd = point_cld[mask2]
    # back_pcd = point_cld

    return fore_pcd, back_pcd # [num_points, 6]


def init_pcd_generate(path, output_dir):
    # create cam_info
    cam_info = readCamerasFromTransforms(path, "transforms.json", white_background=False)

    # load intrinsics
    with open(os.path.join(path, "transforms.json"), "r") as f:
        data = json.load(f)
        FX = torch.tensor(data["fl_x"], dtype=torch.float32).cuda()
        FY = torch.tensor(data["fl_y"], dtype=torch.float32).cuda()
        CX = torch.tensor(data["cx"], dtype=torch.float32).cuda()
        CY = torch.tensor(data["cy"], dtype=torch.float32).cuda()

    depth_imgs = {}  # add depth info
    file_name = 'metric3d_depth_result'
    # file_name = 'realsense_depths'
    # file_name = 'depth'
    depth_path = os.path.join(output_dir, file_name)
    # natural sorting
    file_list = sorted(os.listdir(depth_path), key=lambda x: int(x.split('_')[1].split('.')[0]))

    for i, filename in enumerate(file_list):
        print(filename)
        depth_img = cv2.imread(os.path.join(depth_path, filename), cv2.IMREAD_ANYDEPTH) / 1000
        depth_img = torch.tensor(depth_img, dtype=torch.float32).cuda()
        print(f"depth {i} max: {torch.max(depth_img)} min: {torch.min(depth_img)}")
        name, _ = os.path.splitext(filename)
        id = name.split('_')[-1]
        depth_imgs[id] = depth_img

    w2c = torch.eye(4, dtype=torch.float32).cuda()
    pcds = o3d.geometry.PointCloud()
    back_pcds = o3d.geometry.PointCloud()
    fore_pcds = o3d.geometry.PointCloud()
    fore_pcd_o3d = o3d.geometry.PointCloud()
    back_pcd_o3d = o3d.geometry.PointCloud()
    for i in tqdm(range(len(cam_info))):
        rgb = ToTensor()(cam_info[i].image).cuda()
        num = int(cam_info[i].image_name.split('_')[1])
        print(f"num: {num}")
        depth = depth_imgs[f'{num}'].cuda()
        w2c = torch.tensor(getWorld2View2(cam_info[i].R, cam_info[i].T)).cuda()
        fore_pcd_np, back_pcd_np = get_pointcloud(rgb, depth, w2c, FX, FY, CX, CY, transform_pts=True)
        fore_pcd_np = fore_pcd_np.cpu().numpy()
        back_pcd_np = back_pcd_np.cpu().numpy()
        # fore_pcd_o3d.points = o3d.utility.Vector3dVector(fore_pcd_np[:, :3])
        # fore_pcd_o3d.colors = o3d.utility.Vector3dVector(fore_pcd_np[:, 3:])
        back_pcd_o3d.points = o3d.utility.Vector3dVector(back_pcd_np[:, :3])
        back_pcd_o3d.colors = o3d.utility.Vector3dVector(back_pcd_np[:, 3:])
        back_pcd_o3d = back_pcd_o3d.voxel_down_sample(voxel_size=0.02)
        # fore_pcds += fore_pcd_o3d
        back_pcds += back_pcd_o3d

    # directly extract pcds from depth
    # back_pcds = back_pcds.voxel_down_sample(voxel_size=0.02)
    # pcds = back_pcds

    fore_pcds_path = os.path.join(output_dir, 'foreground_pcd.ply')
    if os.path.exists(fore_pcds_path):
        fore_pcds = o3d.io.read_point_cloud(fore_pcds_path)
        fore_pcds.paint_uniform_color([0,0,0])
        # fore_pcds = fore_pcds.voxel_down_sample(voxel_size=0.02)
        print(len(fore_pcds.points))
        print(len(back_pcds.points))
        pcds = back_pcds + fore_pcds
        # pcds = fore_pcds
    print( len(pcds.points))
    o3d.io.write_point_cloud(os.path.join(output_dir, f'merged_pcd.ply'), pcds)

