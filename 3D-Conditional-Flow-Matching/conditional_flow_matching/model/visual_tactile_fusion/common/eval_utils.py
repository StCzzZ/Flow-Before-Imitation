import torch
import kaolin
import numpy as np
import math
from scipy.spatial import distance
import trimesh


def compute_iou(occ1, occ2, threshold=0.5):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= threshold)
    occ2 = (occ2 >= threshold)
    
    # threshold = np.mean(occ2)
    # occ1 = (occ1 >= threshold)
    # occ2 = (occ2 >= threshold)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou


def cal_square_loss(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(y - y_pred))


def cal_cmap(hand_pcd: torch.Tensor, obj_pcd:torch.Tensor) -> torch.Tensor:
    """
    :param hand_pcd: the batch point cloud of hand with shape (B, N1, 3)
    :param obj_pcd: the batch point cloud of object with shape (B, N2, 3)
    :return: contact_map with shape (B, N2)
    """
    B, N1, _ = hand_pcd.shape
    _, N2, _ = obj_pcd.shape

    d_1 = torch.tile(torch.sum(obj_pcd ** 2, dim=2).unsqueeze(2), (1, 1, N1))
    d_2 = torch.tile(torch.sum(hand_pcd ** 2, dim=2).unsqueeze(1), (1, N2, 1))
    d_3 = torch.einsum('ijk,ilk->ijl', obj_pcd, hand_pcd)
    d = d_1 + d_2 - 2 * d_3
    nearest_dist = torch.min(d, dim=2)[0]
    nearest_dist = torch.sqrt(nearest_dist)
    cmap = 1 - 2 * (torch.sigmoid(2 * nearest_dist) - 0.5)
    return cmap

def cal_loss_h(hand_pcd: torch.Tensor, obj_pcd:torch.Tensor) -> torch.Tensor:
    """
    :param hand_pcd: the batch point cloud of hand with shape (B, N1, 3)
    :param obj_pcd: the batch point cloud of object with shape (B, N2, 3)
    :return: hand-centric object loss with shape (B)
    """
    B, N1, _ = hand_pcd.shape
    _, N2, _ = obj_pcd.shape

    d_1 = torch.tile(torch.sum(obj_pcd ** 2, dim=2).unsqueeze(2), (1, 1, N1))
    d_2 = torch.tile(torch.sum(hand_pcd ** 2, dim=2).unsqueeze(1), (1, N2, 1))
    d_3 = torch.einsum('ijk,ilk->ijl', obj_pcd, hand_pcd)
    d = d_1 + d_2 - 2 * d_3
    nearest_dist = torch.min(d, dim=2)[0]

    nearest_dist = torch.sqrt(nearest_dist)

    loss_h = torch.mean(torch.where(nearest_dist < 0.01, nearest_dist, torch.zeros_like(nearest_dist)))
    if torch.isnan(loss_h):
        print(torch.sum(nearest_dist < 0.01))

    return loss_h

def cal_pen_loss(vertices_batch: torch.Tensor, faces_batch: torch.Tensor,
                 Tbase: torch.Tensor,
                 obj_pcd: torch.Tensor) -> torch.Tensor:
    """
    Calculate the penetration depth using hand mesh and object pcd
    :param gripper:
    :param Tbase: wrist transformation matrix, (batch_size, 4, 4)
    :param joints: joint parameter, (batch_size, gripper_dof)
    :param obj_pcd: object point cloud, (num_point, 3)
    :return: pen: max penetration depth, (batch_size, )
    """
    device = vertices_batch.device
    batch_size = vertices_batch.shape[0]

    pen = torch.zeros((batch_size, ), dtype=torch.float32).to(device)
    pen_num = torch.zeros((batch_size, ), dtype=torch.float32).to(device)
    for i in range(batch_size):
        vertices = vertices_batch[i] # (batch_size, num_vertices, 3)
        faces = faces_batch[i]      # (num_faces, 3)
        face_vertices = kaolin.ops.mesh.index_vertices_by_faces(vertices, faces)
        dist, face_indices, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(obj_pcd, face_vertices)
        dist_sign = kaolin.ops.mesh.check_sign(vertices, faces, obj_pcd)
        dist_tmp = torch.zeros_like(dist)
        dist_tmp[dist_sign] += dist[dist_sign]
        pen_num = pen_num + torch.sum(dist_sign)
        pen = pen + torch.sum(dist_tmp, dim = 1)
    pen_loss = torch.sum(pen / (pen_num + 1))
    return pen_loss

def cal_kld(means: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + log_var - means.pow(2) - log_var.exp())
