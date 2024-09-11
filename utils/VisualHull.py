import cv2
import os
import json
import open3d as o3d
import numpy as np
from tqdm import tqdm
from itertools import product
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
from pathlib import Path
from PIL import Image
from utils.readCam import readCamerasFromTransforms

# Define a voxel grid which has the 3D locations of each voxel which can then be projected onto each image
def InitializeVoxels(xlim, ylim, zlim, voxel_size):
  voxels_number = [1, 1, 1]
  voxels_number[0] = np.abs(xlim[1]-xlim[0]) / voxel_size[0]
  voxels_number[1] = np.abs(ylim[1]-ylim[0]) / voxel_size[1]
  voxels_number[2] = np.abs(zlim[1]-zlim[0]) / voxel_size[2]
  voxels_number_act = np.array(voxels_number).astype(int) + 1
  total_number = np.prod(voxels_number_act)

  voxel = np.ones((int(total_number), 4))

  sx = xlim[0]
  ex = xlim[1]
  sy = ylim[0]
  ey = ylim[1]
  sz = zlim[0]
  ez = zlim[1]

  if(ex > sx):
    x_step = voxel_size[0]
  else:
    x_step = -voxel_size[0]

  if(ey > sy):
    y_step = voxel_size[1]
  else:
    y_step = -voxel_size[1]

  if(sz > ez):
    z_step = voxel_size[2]
  else:
    z_step = -voxel_size[2]

  voxel3Dx, voxel3Dy, voxel3Dz = np.meshgrid(np.linspace(sx, ex, voxels_number_act[0]), np.linspace(sy, ey, voxels_number_act[1]),
    np.linspace(ez, sz, voxels_number_act[2]))
  
  l = 0
  for z in np.linspace(ez, sz, voxels_number_act[2]):
    for x in np.linspace(sx, ex, voxels_number_act[0]):
      for y in np.linspace(sy, ey, voxels_number_act[1]):
        voxel[l] = [x, y, z, 1] 
        l=l+1

  return voxel, voxel3Dx, voxel3Dy, voxel3Dz, voxels_number


def ConvertVoxelList2Voxel3D(voxels_number, voxel_size, voxel):
  sx = -(voxels_number[0] / 2) * voxel_size[0]
  ex = voxels_number[0] / 2 * voxel_size[0]

  sy = -(voxels_number[1] / 2) * voxel_size[1]
  ey = voxels_number[1] / 2 * voxel_size[1]
  sz = 0
  ez = voxels_number[2] * voxel_size[2]
  voxels_number = np.array(voxels_number).astype(np.int32)
  voxel3D = np.zeros((voxels_number[1] + 1, voxels_number[0] + 1, voxels_number[2] + 1))

  l = 0
  z1 = 0
  for z in np.arange(ez, sz, -voxel_size[2]):
      x1 = 0
      for x in np.arange(sx, ex, voxel_size[0]):
          y1 = 0
          for y in np.arange(sy, ey, voxel_size[1]):
              voxel3D[y1, x1, z1] = voxel[l, 3]
              l = l + 1
              y1 = y1 + 1
          x1 = x1 + 1
      z1 = z1 + 1

  return voxel3D


def VisualHull(path, error=5):
    # create cam_info
    cam_info = readCamerasFromTransforms(path, "transforms.json", white_background=False)

    # load intrinsics
    with open(os.path.join(path, "transforms.json"), "r") as f:
        data = json.load(f)
        FX = np.float32(data["fl_x"])
        FY = np.float32(data["fl_y"])
        CX = np.float32(data["cx"])
        CY = np.float32(data["cy"])

    M = []
    camera_pose = []
    K = np.eye(3, dtype=np.float32)
    K[0,0] = FX
    K[1,1] = FY
    K[0,2] = CX
    K[1,2] = CY
    w2c = np.eye(4, dtype=np.float32)
    for i in range(len(cam_info)): 
      print(cam_info[i].image_name)
      R = (cam_info[i].R).T
      t = (cam_info[i].T).reshape(3, 1)

      w2c[:3, :3] = R
      w2c[:3, 3] = cam_info[i].T
      c2w = np.linalg.inv(w2c)
      camera_pose.append(c2w[:3, 3])
      M.append(np.matmul(K, np.concatenate([R, t], axis=1)))

    camera_center = np.mean(camera_pose, axis=0)
    print('camera_center:', camera_center)
    # Reading images
    imgs = []
    file_name = 'masks'
    mask_path = os.path.join(path, file_name)
    file_list = sorted(os.listdir(mask_path), key=lambda x: x.zfill(10))
    for i in tqdm(range(len(cam_info))):
        # if i % 10 != 0:
        #     continue
        filename = '{}.png'.format(cam_info[i].image_name)
        mask_img = cv2.imread(os.path.join(mask_path, filename), cv2.IMREAD_UNCHANGED)
        if mask_img.ndim == 3:
            mask_img = mask_img[:, :, 0]
        imgs.append(np.array(mask_img/255))
    imgs = np.array(imgs).transpose(1, 2, 0)

    voxel_size = [0.005, 0.005, 0.005] # size of each voxel
    # The dimension limits
    xmin = camera_center[0] - 0.5
    xmax = camera_center[0] + 0.5
    ymin = camera_center[1] - 0.5
    ymax = camera_center[1] + 0.5
    zmin = camera_center[2] - 0.5
    zmax = camera_center[2] + 0.5
    xlim = [xmin, xmax]
    ylim = [ymin, ymax]
    zlim = [zmin, zmax]

    voxels, voxel3Dx, voxel3Dy, voxel3Dz, voxels_number = InitializeVoxels(xlim, ylim, zlim, voxel_size)

    object_points3D = np.copy(voxels).T
    voxels[:, 3] = 0 # making the fourth variable of each voxel 0

    proj = []

    for i in tqdm(range(len(cam_info))):
          # CAMERA PARAMETERS
          M_ = M[i]
          # PROJECTION TO THE IMAGE PLANE
          points2D = np.matmul(M_, object_points3D)
          points2D = np.floor(points2D / points2D[2, :] + 1e-6).astype(np.int32)
          points2D[np.where(points2D < 0)] = 0; # check for negative image coordinates

          img_size = (imgs).shape
          ind1 = np.where(points2D[1, :] >= img_size[0]) # check for out-of-bounds (width) coordinate
          points2D[:, ind1] = 0
          ind1 = np.where(points2D[0, :] >= img_size[1]) # check for out-of-bounds (height) coordinate
          points2D[:, ind1] = 0
          count_nonzero = np.count_nonzero(points2D.T[:, 0])
          print(f'{count_nonzero} non-zero points')
          # ACCUMULATE THE VALUE OF EACH VOXEL IN THE CURRENT IMAGE
          voxels[:, 3] += imgs[:, :, i].T[points2D.T[:, 0], points2D.T[:, 1]]

          proj.append(points2D)

    error_amount = error
    maxv = np.max(voxels[:, 3])
    iso_value = maxv-np.round(((maxv)/100)*error_amount)-0.5
    print('max number of votes:' + str(maxv))
    print('threshold for marching cube:' + str(iso_value))

    voxel3D = ConvertVoxelList2Voxel3D(np.array(voxels_number), voxel_size, voxels)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Extract the coordinates of non-zero voxels
    occupied_pt = voxels[:, 3] > iso_value
    filtered_voxels  = voxels[occupied_pt]
    x = filtered_voxels[:, 0]
    y = filtered_voxels[:, 1]
    z = filtered_voxels[:, 2]
    # save as pcd
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_voxels[:, :3])
    o3d.io.write_point_cloud(f'{path}/foreground_pcd.ply', pcd)
    # Plot the voxels
    ax.scatter(x, y, z, c='r', marker='o')
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(f'{path}/voxels.png', dpi=600, bbox_inches='tight', pad_inches=0.05)
