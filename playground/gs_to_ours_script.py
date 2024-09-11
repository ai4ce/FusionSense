import os
import numpy as np
import open3d as o3d
import json
import re
from pathlib import Path


def read_txt_array(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        array = [int(value) for value in content.split(',')]
    return array

if '__name__' == '__main__':
    # Create a all-one mask for point cloud
    dataset_dir = Path('./datasets/touchgs')
    patch_dir = dataset_dir / Path('./tactile/patch')
    mask_dir = dataset_dir / Path('./tactile/mask')
    normal_dir = dataset_dir / Path('./tactile/normal')

    in_transform_path = "transforms_train.json"
    out_transform_path = "gelsight_transform.json"
    # annotated = ["59", "119", "134", "141", "154"]
    annotated = read_txt_array(dataset_dir / "train.txt")

    os.makedirs(mask_dir, exist_ok=True)
    count = 0
    in_frames = json.load(open(in_transform_path, "r"))
    out_frames = {
        "frames" : [],
        "applied_transform" : [
            [
                1.0,
                0.0,
                0.0,
                0.0
            ],
            [
                0.0,
                1.0,
                0.0,
                0.0
            ],
            [
                0.0,
                0.0,
                1.0,
                0.0
            ]
        ]
    }

    for in_frame in in_frames["frames"]:
        match = re.search(r'tr_(\d+)\.png', in_frame["file_path"])
        if match and int(match.group(1)) in annotated:
            count += 1
            id = match.group(1)
            out_frame = {}
            out_frame["colmap_im_id"] = id
            out_frame["patch_path"] = f'{patch_dir}/tr_{id}.ply'
            out_frame["mask_path"] = f'{mask_dir}/tr_{id}.npy'
            out_frame["normal_path"] = f'{normal_dir}/tr_{id}.npy'
            out_frame["transform_matrix"] = in_frame["transform_matrix"]
            out_frame["file_path"] = in_frame["file_path"]
        
            # Since the patch data are already masked, create an all-ones mask and save as .npy in the mask directory
            pcd = o3d.io.read_point_cloud(f"{patch_dir}/tr_{id}.pcd")
            num_points = np.asarray(pcd.points).shape[0]
            mask = np.ones(num_points, dtype=np.int32)
            np.save(out_frame["mask_path"], mask)
            out_frames["frames"].append(out_frame)
        
    with open(out_transform_path, 'w') as f:
        json.dump(out_frames, f)

    print(f"{count} masks have been generated and saved.")