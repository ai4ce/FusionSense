import numpy as np
from sklearn.cluster import DBSCAN
import open3d as o3d
import open3d.core as o3c
import random
import torch
import os

def dbscan_cluster_centers(high_grad_points, path, eps=0.01, min_samples=15):
    """
    DBSCAN-based clustering function, takes in a set of points and outputs the coordinates of cluster centers.
    
    Parameters:
    - points: input point set, shape is (n_samples, n_features)
    - eps: DBSCAN's epsilon parameter, defines the distance threshold for neighborhood
    - min_samples: DBSCAN's min_samples parameter, defines the minimum number of neighbors for a core point
    
    Returns:
    - cluster_centers: coordinates of each cluster center
    """
    high_grad_points = high_grad_points.detach().cpu().numpy()
    points = high_grad_points[:, :3]

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    
    labels = db.labels_
    cluster_centers = []
    ave_grads = []
    points_cluster_ctrs = np.zeros((points.shape[0], 3))
    points_ranks = np.zeros((points.shape[0], 1)) 
    colors = np.zeros((points.shape[0], 3)) 

    unique_labels = set(labels) - {-1}
    for label in unique_labels:
        cluster_points = points[labels == label]
        center = np.mean(cluster_points, axis=0)
        cluster_centers.append(center)

        gradients = high_grad_points[:, :3][labels == label]
        ave_grad = np.mean(gradients, axis=0)
        ave_grads.append(ave_grad)

    sorted_indices = sorted(range(len(ave_grads)), key=lambda i: np.linalg.norm(ave_grads[i]))
    ranks = {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}

    color_map = {rank: [random.random(), random.random(), random.random()] for rank in ranks.values()}

    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        cluster_rank = ranks[list(unique_labels).index(label)]
        points_ranks[cluster_indices] = cluster_rank

        cluster_color = color_map[cluster_rank]
        colors[cluster_indices] = cluster_color

        cluster_points = points[labels == label]
        points_cluster_ctrs[cluster_indices] = cluster_points

    features = np.zeros((points.shape[0], 4)) # features: [rank, x, y, z]
    features[:, 0] = points_ranks.reshape(-1)
    features[:, 1:] = points_cluster_ctrs

    # # Save cluster centers as a npy file
    # cluster_centers = np.array(cluster_centers)
    # save_path_npy = os.path.join(path, 'cluster_centers.npy')
    # np.save(save_path_npy, cluster_centers)
    # print(f"Cluster centers saved as high_grad_pts.npy")
    
    # Create a PointCloud and assign colors
    pcd = o3d.t.geometry.PointCloud(points)
    # pcd.points = o3c.Tensor(points)
    pcd.point.colors = o3c.Tensor(colors)
    pcd.point.cluster_ctrs_x = o3c.Tensor(points_cluster_ctrs[:, 0].reshape(-1, 1))
    pcd.point.cluster_ctrs_y = o3c.Tensor(points_cluster_ctrs[:, 1].reshape(-1, 1))
    pcd.point.cluster_ctrs_z = o3c.Tensor(points_cluster_ctrs[:, 2].reshape(-1, 1))
    pcd.point.ranks = o3c.Tensor(points_ranks)

    # Save as a PCD file
    save_path_pcd = os.path.join(path, 'high_grad_pts.pcd')
    save_path_ascii_pcd = os.path.join(path, 'high_grad_pts_ascii.pcd')
    print(save_path_pcd)
    o3d.t.io.write_point_cloud(save_path_pcd, pcd)
    o3d.t.io.write_point_cloud(save_path_ascii_pcd, pcd, write_ascii=True)
    print(f"Colored point cloud saved as high_grad_pts.pcd")
    
    return np.array(cluster_centers)