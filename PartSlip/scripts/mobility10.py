import open3d as o3d
import numpy as np
import os
import json
import torch
from mobility_transform import rotate_module

def rotate_part(pts, pointIndicator, joint_dir, joint_pos, angle):
    """
    rotate the moving part along (joint_dir, joint_pos) with angle.
    """
    part_pts = pts[pointIndicator == 1]
    part_pts -= joint_pos
    motion_quat = np.hstack(
        [np.cos(angle / 360 * 2 * 3.14 / 2), np.sin(angle / 360 * 2 * 3.14 / 2) * joint_dir])
    part_pts = \
        rotate_module(torch.from_numpy(part_pts).view(
            1, -1, 3), torch.from_numpy(motion_quat).view(1, 1, 4)).numpy()[0]
    part_pts += joint_pos
    pts[pointIndicator == 1] = part_pts
    return pts


def translate_part(pts, pointIndicator, joint_dir, amount):
    """
    translate the moving part along the translation axis with amount.
    """
    part_pts = pts[pointIndicator == 1]
    part_pts += (joint_dir * amount)
    pts[pointIndicator == 1] = part_pts
    return pts

for category in os.listdir('dataset_classified_2'):
    category_directory = os.path.join('dataset_classified_2', category)
    for id in os.listdir(category_directory):
        file_path = os.path.join(category_directory, id)

        if os.path.isdir(os.path.join(file_path, 'point_sample')):
            print(file_path)

            for step in range(10):
                motion_path = os.path.join(file_path, 'mobility_v2.json')
                motion_data = json.load(open(motion_path, 'r'))

                ply_path = os.path.join(file_path, 'point_sample', 'sample-points-all-pts-nor-rgba-10000.ply')
                point_cloud = o3d.io.read_point_cloud(ply_path)
                shape_pc = np.asarray(point_cloud.points).copy()
                # o3d.visualization.draw_geometries([point_cloud])

                label_path = motion_path = os.path.join(file_path, 'point_sample/sample-points-all-label-10000.txt')
                label_file = open(label_path, 'r')
                labels = [line.strip() for line in label_file.readlines()]
                shape_pc_ids = np.array(labels, dtype=int)


                for joint_part in motion_data:
                    if joint_part['joint'] == 'hinge':
                        joint_position = np.asarray(joint_part['jointData']['axis']['origin'])
                        joint_direction = np.asarray(joint_part['jointData']['axis']['direction']).astype(np.float64)

                        joint_limit = joint_part['jointData']['limit']
                        angle_min = min(joint_limit['a'], joint_limit['b'])
                        angle_max = max(joint_limit['a'], joint_limit['b'])
                        # angle = np.random.uniform(low=min(joint_limit['a'], joint_limit['b']), high=max(joint_limit['a'], joint_limit['b']))
                        movePointIds = []
                        for part in joint_part['parts']:
                            movePointIds.append(part['id'])
                        pointIndicator = np.zeros(len(shape_pc_ids)).astype(int)
                        for n in movePointIds:
                            pointIndicator[shape_pc_ids == int(n)] = 1
                        shape_pc = rotate_part(shape_pc, pointIndicator, joint_direction, joint_position, angle_min + (angle_max - angle_min) / 9 * step)
                    
                    elif joint_part['joint'] == 'slider':
                        joint_direction = np.asarray(joint_part['jointData']['axis']['direction']).astype(np.float64)
                        joint_limit = joint_part['jointData']['limit']
                        amount_min = min(joint_limit['a'], joint_limit['b'])
                        amount_max = max(joint_limit['a'], joint_limit['b'])
                        # angle = np.random.uniform(low=min(joint_limit['a'], joint_limit['b']), high=max(joint_limit['a'], joint_limit['b']))
                        movePointIds = []
                        for part in joint_part['parts']:
                            movePointIds.append(part['id'])
                        pointIndicator = np.zeros(len(shape_pc_ids)).astype(int)
                        for n in movePointIds:
                            pointIndicator[shape_pc_ids == int(n)] = 1
                        shape_pc = translate_part(shape_pc, pointIndicator, joint_direction, amount_min + (amount_max - amount_min) / 9 * step)

                point_cloud.points = o3d.utility.Vector3dVector(shape_pc)
                # o3d.visualization.draw_geometries([point_cloud])
                output_ply_path = os.path.join(file_path, 'point_sample', str(step)+'_motion.ply')
                o3d.io.write_point_cloud(output_ply_path, point_cloud)
