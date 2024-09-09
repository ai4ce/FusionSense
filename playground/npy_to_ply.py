import numpy as np
import open3d as o3d
import cv2

def npy_to_ply(npy_file, ply_file):
    """
    Convert a .npy file containing point cloud data to a .ply file.

    :param npy_file: Path to the input .npy file.
    :param ply_file: Path to the output .ply file.
    """
    # Load the point cloud data from the .npy file
    point_cloud_data = np.load(npy_file)

    # Check if the data is of shape (N, 3) or (N, 6) (optional colors/normals)
    if point_cloud_data.shape[1] not in [3, 6]:
        raise ValueError("The .npy file should contain a Nx3 or Nx6 array.")

    # Create an open3d point cloud object
    pcd = o3d.geometry.PointCloud()

    # Assign points to the point cloud object
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])

    # If the array contains color information (Nx6), assign colors
    if point_cloud_data.shape[1] == 6:
        pcd.colors = o3d.utility.Vector3dVector(point_cloud_data[:, 3:])

    # Save the point cloud as a .ply file
    o3d.io.write_point_cloud(ply_file, pcd)
    print(f"Point cloud saved to {ply_file}")

if __name__ == "__main__":
    # Example usage
    npy_file = "datasets/BlackBunny/tactile/normal/0.npy"  # Replace with your .npy file path
    normal_data = np.load(npy_file)
    print(normal_data)
    print(normal_data.shape)
    print(np.max(normal_data))
    # png_file = "datasets/BlackBunny/tactile/image/0.png"
    # pic = cv2.imread(png_file, cv2.IMREAD_UNCHANGED)
    # print(pic/255.)
    # print(np.max(pic)/255.)
    
    # ply_file = "datasets/touch-rabbit/tr_0.ply" # Replace with your desired .ply file path
    
    # npy_to_ply(npy_file, ply_file)