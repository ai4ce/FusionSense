import json
import os
import random
import typing
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Literal, Optional, Type

import numpy as np
import open3d as o3d
import torch
import trimesh
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from torch.cuda.amp.grad_scaler import GradScaler

from dn_splatter.data.mushroom_utils.eval_faro import depth_eval_faro
from dn_splatter.dn_model import DNSplatterModelConfig
from dn_splatter.metrics import PDMetrics
from dn_splatter.utils import camera_utils
from dn_splatter.utils.utils import gs_render_dataset_images, ns_render_dataset_images
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
    VanillaDataManager,
)
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    DDP,
    Model,
    VanillaPipeline,
    VanillaPipelineConfig,
    dist,
)
from nerfstudio.utils import profiler
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class DNSplatterLocalPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: DNSplatterLocalPipeline)
    datamanager: DataManagerConfig = field(default_factory=lambda: DataManagerConfig())
    model: ModelConfig = field(default_factory=lambda: DNSplatterModelConfig())
    experiment_name: str = "experiment"
    """Experiment name for saving metrics and rendered images to disk"""
    skip_point_metrics: bool = True
    """Skip evaluating point cloud metrics"""
    num_pd_points: int = 1_000_000
    """Total number of points to extract from train/eval renders for pointcloud reconstruction"""
    save_train_images: bool = False
    """saving train images to disc"""

class DNSplatterLocalPipeline(VanillaPipeline):
    def __init__(
        self,
        config: DNSplatterLocalPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
        )
        self.datamanager.to(device)
        seed_pts = None
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata  # type: ignore
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata[ "points3D_xyz" ]  # type: ignore
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata[ "points3D_rgb" ]  # type: ignore
            if "points3D_normals" in self.datamanager.train_dataparser_outputs.metadata:
                normals = self.datamanager.train_dataparser_outputs.metadata[
                    "points3D_normals"
                ]  # type: ignore

                # initialize patch points 
                if "touch_patches" in self.datamanager.train_dataparser_outputs.metadata:
                    touch_patches = self.datamanager.train_dataparser_outputs.metadata[
                        "touch_patches"    
                    ] # type: ignore
                    # next 
                    current_touch_index = 0
                    touch_patch = touch_patches[current_touch_index]
                    touch_pts = torch.cat((pts, touch_patch["points_xyz"])).detach() # disable position gradient
                    touch_pts_rgb = torch.cat((pts_rgb, touch_patch["points_rgb"]))
                    touch_pts_normals = torch.cat((normals, touch_patch["normals"]))

                    max_xyz = torch.max(touch_pts, axis=0).values
                    min_xyz = torch.min(touch_pts, axis=0).values
                    diag_xyz = (max_xyz - min_xyz) * 0.1
                    min_aabb = min_xyz-diag_xyz
                    max_aabb = max_xyz+diag_xyz
                    mask_x = (pts[:, 0] >= min_aabb[0]) & (pts[:, 0] <= max_aabb[0])
                    mask_y = (pts[:, 1] >= min_aabb[1]) & (pts[:, 1] <= max_aabb[1])
                    mask_z = (pts[:, 2] >= min_aabb[2]) & (pts[:, 2] <= max_aabb[2])
                    aabb_mask = mask_x & mask_y & mask_z
                    
                    # only train points in aabb, so detach other points in the background
                    bg_pts = pts[~aabb_mask].detach()
                    bg_pts_rgb = pts_rgb[~aabb_mask].detach()
                    bg_pts_normal = normals[~aabb_mask].detach()

                    fused_pts = torch.cat((pts[aabb_mask], bg_pts, touch_pts))
                    fused_pts_rgb = torch.cat((pts_rgb[aabb_mask], bg_pts_rgb, touch_pts_rgb))
                    fused_pts_normals = torch.cat((normals[aabb_mask], bg_pts_normal, touch_pts_normals))

                    '''For local training, rgb loss should not be counted due to '''
                    seed_pts = (fused_pts, fused_pts_rgb, fused_pts_normals)
                else:
                    seed_pts = (pts, pts_rgb, normals)
            else:
                seed_pts = (pts, pts_rgb)
        self.datamanager.to(device)
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            seed_points=seed_pts,
        )
        self.model.to(device)
        self.pd_metrics = PDMetrics() # acc and cmp error

    @profiler.time_function
    def get_average_eval_image_metrics(
        self,
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
    ):
        metrics_dict_list = []
        points_eval = []
        colors_eval = []
        points_train = []
        colors_train = []
        self.eval()
        metrics_dict = {}
        self.train()
        return metrics_dict
