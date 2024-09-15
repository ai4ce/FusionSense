import os
import json
import open3d as o3d
import numpy as np
import time
import copy
import torch
from scipy.spatial import cKDTree
 
def Chamfer_Distance(pcd_cad, pcd_real):
    source_pcd = np.asarray(pcd_cad.points)
    target_pcd = np.asarray(pcd_real.points)
    tree1 = cKDTree(source_pcd)
    tree2 = cKDTree(target_pcd)
    dist1, _ = tree1.query(target_pcd)
    dist2, _ = tree2.query(source_pcd)
    chamfer_dist = np.mean(dist1**2) + np.mean(dist2**2)
    return chamfer_dist * 1e3 

def FPFH_Compute(pcd):
    radius_normal = 0.01
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = 0.02
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_fpfh
 
def execute_global_registration(source, target, source_fpfh,
                                target_fpfh):
    distance_threshold = 0.01
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.1),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

def Icp_preprocessing(mesh, pcd_real):
    # Sample points from the mesh to create a point cloud
    pcd_cad = mesh.sample_points_poisson_disk(number_of_points=500000)
    # change cad mm scale to meter
    points = np.asarray(pcd_cad.points)
    points = points / 1000.0
    pcd_cad.points = o3d.utility.Vector3dVector(points)

    # RANSAC registration
    pcd_cad_fpfh=FPFH_Compute(pcd_cad)
    pcd_real_fpfh=FPFH_Compute(pcd_real)
    result_ransac = execute_global_registration(pcd_cad, pcd_real, pcd_cad_fpfh, pcd_real_fpfh)
    Tr = result_ransac.transformation
    # Apply the RANSAC transformation to the CAD point cloud
    pcd_cad.transform(Tr)

    initial_transform = np.eye(4)
    icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
    max_iteration = 1000000   # Maximum number of iterations
    )

    pcd_cad.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pcd_real.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    # Perform ICP
    icp_result = o3d.pipelines.registration.registration_icp(
        pcd_cad, pcd_real, max_correspondence_distance=0.1, init=initial_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=icp_criteria
    )

    # Print the transformation matrix
    print("ICP Transformation Matrix:")
    print(icp_result.transformation)

    # Apply the transformation to the CAD point cloud
    pcd_cad.transform(icp_result.transformation)
    return pcd_cad

def local_CD(center, pcd_cad, pcd_real, radius=0.01):
    points1 = np.asarray(pcd_cad.points)
    distances = np.linalg.norm(points1 - center, axis=1)
    mask = distances > radius 
    filtered_points = points1[mask]
    local_pcd_cad = o3d.geometry.PointCloud()
    local_pcd_cad.points = o3d.utility.Vector3dVector(filtered_points)

    points2 = np.asarray(pcd_real.points)
    distances = np.linalg.norm(points2 - center, axis=1)
    mask = distances > radius
    filtered_points = points2[mask]
    local_pcd_real = o3d.geometry.PointCloud()
    local_pcd_real.points = o3d.utility.Vector3dVector(filtered_points)

    local_chamfer_dist = Chamfer_Distance(local_pcd_cad, local_pcd_real)
    return local_chamfer_dist

def touch_CD_eval(base_dir, pcd_cad, pcd_real):
    with open(os.path.join(base_dir, "gelsight_transform.json")) as f:
        gel_transforms = json.load(f)
    touch_centers = []
    for frame in gel_transforms["frames"]:
        if "transform_matrix" in frame:
            touch_center = np.array(frame["transform_matrix"])[:3,3]
            touch_centers.append(touch_center)

    local_chamfer_dist = []
    for center in touch_centers:
        local_chamfer_dist.append(local_CD(center, pcd_cad, pcd_real))
    avg_local_chamfer_dist = np.mean(local_chamfer_dist)
    print(f"Average Local Chamfer Distance: {avg_local_chamfer_dist}")
    return np.array(avg_local_chamfer_dist)

def chamfer_eval(base_dir, mesh_dir):
    pcd_real = o3d.io.read_point_cloud(os.path.join(mesh_dir, "after_clean_points_surface_level_0.3_closest_gaussian_touch.ply"))
    cad_mesh = o3d.io.read_triangle_mesh(os.path.join(base_dir, "stanford_bunny.stl"))
    pcd_cad = Icp_preprocessing(cad_mesh, pcd_real)
    o3d.io.write_point_cloud(os.path.join(mesh_dir, "pcd_cad.ply"), pcd_cad)

    # # global chamfer distance
    # dist = Chamfer_Distance(pcd_cad, pcd_real)
    # print(f"Chamfer Distance: {dist}")

    # local chamfer distance
    dist = touch_CD_eval(base_dir, pcd_cad, pcd_real)

    with open(os.path.join(mesh_dir, "chamfer_distance_eval.json"), "w") as f:
        json.dump({"chamfer_distance": dist}, f, indent=4)

# if __name__ == "__main__":
#     pcd_real = o3d.io.read_point_cloud("outputs/transparent_bunny/MESH/after_clean_points_surface_level_0.3_closest_gaussian.ply")
#     cad_mesh = o3d.io.read_triangle_mesh("/home/ks8018/dn-splatter/outputs/blackbunny3/MESH/stanford_bunny.stl")
#     pcd_cad = Icp_preprocessing(cad_mesh, pcd_real)
#     dist = Chamfer_Distance(pcd_cad, pcd_real)

#     o3d.io.write_point_cloud("/home/ks8018/dn-splatter/outputs/transparent_bunny/MESH/pcd_cad.ply", pcd_cad)
#     print(f"Chamfer Distance: {dist}")