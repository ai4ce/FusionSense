
import os
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from dn_splatter.losses import DepthLoss, DepthLossType, TVLoss
from dn_splatter.metrics import DepthMetrics, NormalMetrics, RGBMetrics
from dn_splatter.utils.camera_utils import get_colored_points_from_depth, project_pix
from dn_splatter.utils.knn import knn_sk
from dn_splatter.utils.normal_utils import normal_from_depth_image
from dn_splatter.utils.utils import depth_to_colormap

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")
from gsplat import rasterize_gaussians
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
from gsplat.cuda_legacy._wrapper import num_sh_bases
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.splatfacto import (
    RGB2SH,
    SplatfactoModel,
    SplatfactoModelConfig,
    get_viewmat,
)
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE

from dn_splatter.dn_model import DNSplatterModelConfig, DNSplatterModel

def Override(method):
    def wrapper(*args, **kwargs):
        assert method.__name__ in dir(args[0].__class__.__bases__[0]), f"{method.__name__} does not override any method"
        return method(*args, **kwargs)
    return wrapper

class DNSplatterLocalModel(DNSplatterModel):
    @Override
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, Tensor]:
        loss_dict = super().get_loss_dict(
            outputs=outputs, batch=batch, metrics_dict=metrics_dict
        )
        main_loss = loss_dict["main_loss"]
        scale_reg = loss_dict["scale_reg"]

        # Depth Loss
        depth_loss = 0
        if self.config.use_depth_loss:
            depth_loss = 0

        # Normal loss
        normal_loss = 0
        if self.config.use_normal_loss:
            pred_normal = outputs["normal"]

            if "normal" in batch and self.config.normal_supervision == "mono":
                gt_normal = batch["normal"]
            elif self.config.normal_supervision == "depth":
                c2w = self.camera.camera_to_worlds.squeeze(0).detach()
                c2w = c2w @ torch.diag(
                    torch.tensor([1, -1, -1, 1], device=c2w.device, dtype=c2w.dtype)
                )
                gt_normal = normal_from_depth_image(
                    depths=depth_out.detach(),
                    fx=self.camera.fx.item(),
                    fy=self.camera.fy.item(),
                    cx=self.camera.cx.item(),
                    cy=self.camera.cy.item(),
                    img_size=(self.camera.width.item(), self.camera.height.item()),
                    c2w=torch.eye(4, dtype=torch.float, device=depth_out.device),
                    device=self.device,
                    smooth=False,
                )
                gt_normal = gt_normal @ torch.diag(
                    torch.tensor(
                        [1, -1, -1], device=depth_out.device, dtype=depth_out.dtype
                    )
                )
                gt_normal = (1 + gt_normal) / 2
            else:
                CONSOLE.log(
                    "WARNING: You have enabled normal supervision with monocular normals but none were found."
                )
                CONSOLE.log(
                    "WARNING: Remember to first generate normal maps for your dataset using the normals_from_pretrain.py script."
                )
                quit()
            if gt_normal is not None:
                # normal map loss
                normal_loss += torch.abs(gt_normal - pred_normal).mean()
                if self.config.use_normal_cosine_loss:
                    from dn_splatter.metrics import mean_angular_error

                    normal_loss += mean_angular_error(
                        pred=(pred_normal.permute(2, 0, 1) - 1) / 2,
                        gt=(gt_normal.permute(2, 0, 1) - 1) / 2,
                    ).mean()
            if self.config.use_normal_tv_loss:
                normal_loss += self.tv_loss(pred_normal)

        if self.config.two_d_gaussians:
            # loss to minimise gaussian scale corresponding to normal direction
            normal_loss += torch.min(torch.exp(self.scales), dim=1, keepdim=True)[
                0
            ].mean()

        main_loss = (
            depth_loss
            + self.config.normal_lambda * normal_loss
        )
        return {"main_loss": main_loss, "scale_reg": scale_reg}

