import numpy as np
from sklearn.cluster import DBSCAN
import open3d as o3d
import random
import os

def dbscan_cluster_centers(points, path, eps=0.5, min_samples=5):
    """
    DBSCAN-based clustering function, takes in a set of points and outputs the coordinates of cluster centers.
    
    Parameters:
    - points: input point set, shape is (n_samples, n_features)
    - eps: DBSCAN's epsilon parameter, defines the distance threshold for neighborhood
    - min_samples: DBSCAN's min_samples parameter, defines the minimum number of neighbors for a core point
    
    Returns:
    - cluster_centers: coordinates of each cluster center
    """
    if not isinstance(points, np.ndarray):
        points = points.detach().cpu().numpy()
    if points.shape[0] > 100:
        points = points[:100] 

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    
    labels = db.labels_
    cluster_centers = []
    colors = np.zeros((points.shape[0], 3)) 

    unique_labels = set(labels) - {-1}
    for label in unique_labels:
        cluster_points = points[labels == label]
        center = np.mean(cluster_points, axis=0)
        cluster_centers.append(center)

        color = [random.random(), random.random(), random.random()]
        colors[labels == label] = color

    # Save cluster centers as a npy file
    cluster_centers = np.array(cluster_centers)
    np.save(os.path.join(path, 'high_grad_pts.npy'), cluster_centers)
    print(f"Cluster centers saved as high_grad_pts.npy")
    
    # Create a PointCloud and assign colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save as a PCD file
    o3d.io.write_point_cloud(os.path.join(path, 'high_grad_pts.pcd'), pcd)
    print(f"Colored point cloud saved as high_grad_pts.pcd")
    
    return np.array(cluster_centers)