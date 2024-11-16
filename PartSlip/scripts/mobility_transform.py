import json
import numpy as np
import torch


def quat_conjugate(quat):
    # quat = quat.view(-1, 4)

    q0 = quat[:, :, 0]
    q1 = -1 * quat[:, :, 1]
    q2 = -1 * quat[:, :, 2]
    q3 = -1 * quat[:, :, 3]

    q_conj = torch.stack([q0, q1, q2, q3], dim=2)
    return q_conj


def hamilton_product(q1, q2):
    q_size = q1.size()
    # q1 = q1.view(-1, 4)
    # q2 = q2.view(-1, 4)
    inds = torch.LongTensor(
        [0, -1, -2, -3, 1, 0, 3, -2, 2, -3, 0, 1, 3, 2, -1, 0]).view(4, 4)
    q1_q2_prods = []
    for i in range(4):
        # Hack to make 0 as positive sign. add 0.01 to all the values..
        q2_permute_0 = q2[:, :, np.abs(inds[i][0])]
        q2_permute_0 = q2_permute_0 * np.sign(inds[i][0] + 0.01)

        q2_permute_1 = q2[:, :, np.abs(inds[i][1])]
        q2_permute_1 = q2_permute_1 * np.sign(inds[i][1] + 0.01)

        q2_permute_2 = q2[:, :, np.abs(inds[i][2])]
        q2_permute_2 = q2_permute_2 * np.sign(inds[i][2] + 0.01)

        q2_permute_3 = q2[:, :, np.abs(inds[i][3])]
        q2_permute_3 = q2_permute_3 * np.sign(inds[i][3] + 0.01)
        q2_permute = torch.stack(
            [q2_permute_0, q2_permute_1, q2_permute_2, q2_permute_3], dim=2)

        # q1q2_v1 = torch.sum(q1 * q2_permute, dim=2)
        q1q2_v1 = torch.sum(q1 * q2_permute, dim=2,
                            keepdim=True)  
        q1_q2_prods.append(q1q2_v1)

    q_ham = torch.cat(q1_q2_prods, dim=2)
    # q_ham = q_ham.view(q_size)
    return q_ham


def quat_rot_module(points, quats):
    quatConjugate = quat_conjugate(quats) 
    # qvq^(-1)
    mult = hamilton_product(quats, points)
    mult = hamilton_product(mult, quatConjugate)
    return mult[:, :, 1:4]


# points is BxnPx3,  #Bx1x4 quat vectors
def rotate_module(points, quat):
    nP = points.size(1)
    quat_rep = quat.repeat(1, nP, 1)

    zero_points = 0 * points[:, :, 0].clone().view(-1, nP, 1)
    quat_points = torch.cat([zero_points, points], dim=2)

    rotated_points = quat_rot_module(quat_points, quat_rep)  # B x P x 3
    return rotated_points


def transform(pc_path, motion_path, new_pc_path, new_motion_path):
    with h5.File(pc_path, 'r') as f:
        shape_pc = f['pc'][:]
        shape_pc_ids = f['seg'][:]

    motion_data = json.load(open(motion_path, 'r'))
    for part in motion_data:
        part_id = motion_data[part]['leaves']
        inds = []
        for _id in part_id:
            inds.extend(np.where(shape_pc_ids == _id)[0])
        inds = np.asarray(inds)
        if motion_data[part]['joint'] == 'slider':
            direction = np.asarray(
                motion_data[part]['jointData']['axis']['direction'])
            limit = motion_data[part]['jointData']['limit']  # a is the upper limit, b is the lower limit
            amount = np.random.uniform(
                low=min(limit['a'], limit['b']), high=max(limit['a'], limit['b']))
            # translate pts and box
            part_pts_box = np.vstack(
                [shape_pc[inds], np.array(motion_data[part]['box'])])
            part_pts_box += (direction * amount)
            shape_pc[inds] = part_pts_box[:-8]  # new shape_pc
            motion_data[part]['box'] = part_pts_box[-8:].tolist()  # new OBB

        if motion_data[part]['joint'] == 'hinge':
            direction = np.asarray(
                motion_data[part]['jointData']['axis']['direction'])
            position = np.asarray(
                motion_data[part]['jointData']['axis']['origin'])
            limit = motion_data[part]['jointData']['limit']  # a为上限，b为下限
            angle = np.random.uniform(
                low=min(limit['a'], limit['b']), high=max(limit['a'], limit['b']))
            # rotate pts and box
            part_pts_box = np.vstack(
                [shape_pc[inds], np.array(motion_data[part]['box'])])
            part_pts_box -= position
            motion_quat = np.hstack(
                [np.cos(angle / 360 * 2 * 3.14 / 2), np.sin(angle / 360 * 2 * 3.14 / 2) * direction])
            part_pts_box = \
                rotate_module(torch.from_numpy(part_pts_box).view(
                    1, -1, 3), torch.from_numpy(motion_quat).view(1, 1, 4)).numpy()[0]
            part_pts_box += position

            shape_pc[inds] = part_pts_box[:-8]  # new shape_pc
            motion_data[part]['box'] = part_pts_box[-8:].tolist()  # new OBB

    json.dump(motion_data, open(new_motion_path, 'w'))
    with h5.File(new_pc_path, 'w') as f:
        f.create_dataset('pc', data=shape_pc)
        f.create_dataset('seg', data=shape_pc_ids)
