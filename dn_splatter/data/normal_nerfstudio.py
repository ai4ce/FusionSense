from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import numpy as np
import open3d as o3d
import torch
from natsort import natsorted

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.cameras.camera_paths import get_path_from_json
from nerfstudio.data.dataparsers.base_dataparser import (
    DataparserOutputs,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import (
    Nerfstudio,
    NerfstudioDataParserConfig,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
)
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE

MAX_AUTO_RESOLUTION = 1600

# apply a homogenous transformation, then scale
def mut_and_scale(points3D, transform_matrix, scale_factor):
    points3D = (
        torch.cat(
            (
                points3D,
                torch.ones_like(points3D[..., :1]),
            ),
            -1,
        )
        @ transform_matrix.T
    )
    points3D *= scale_factor
    return points3D
    
@dataclass
class NormalNerfstudioConfig(NerfstudioDataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: NormalNerfstudio)
    """target class to instantiate"""
    output_dir: Path = Path("outputs")
    """output dir"""
    load_3D_points: bool = True
    """Whether to load the 3D points from the colmap reconstruction."""
    load_normals: bool = True
    """Set to true to load normal maps"""
    normal_format: Literal["opencv", "opengl"] = "opengl"
    """Which format the normal maps in camera frame are saved in."""
    load_pcd_normals: bool = False
    """Whether to load pcd normals for normal initialisation"""
    
    grad_visualization: bool = False
    """Set to true to enable gradient visualization"""
    load_touches: bool = False
    """Set to true to load normal maps"""
    gel_scale_factor = 6.34e-5 # distance between gel pixels

    orientation_method: Literal['pca', 'up', 'vertical', 'none'] = 'none'
    center_method: Literal['poses', 'focus', 'none'] = 'none'
    auto_scale_poses: bool = True
    scene_scale = 5.0

    load_cameras: bool = False
    camera_path_filename: Path = None

@dataclass
class NormalNerfstudio(Nerfstudio):
    """Nerfstudio DatasetParser"""

    config: NormalNerfstudioConfig
    downscale_factor: Optional[int] = None

    def get_normal_filepaths(self):
        # return glob.glob(f"{self.normal_save_dir}/*.png")
        return natsorted(glob.glob(f"{self.normal_save_dir}/*.png"))
    
    """touch filepath, stored in a 3d pcd"""
    def get_touch_filepaths(self):
        return natsorted(glob.glob(f"{self.touch_data_dir}/patch/*.pcd"))

    def _load_points3D_normals(self, points, colors, transform_matrix: torch.Tensor):
        """Initialise gaussian scales/rots with normals predicted from pcd"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.numpy())
        pcd.colors = o3d.utility.Vector3dVector(colors.numpy())
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.normalize_normals()
        points3D_normals = torch.from_numpy(np.asarray(pcd.normals, dtype=np.float32))
        points3D_normals = (
            torch.cat(
                (points3D_normals, torch.ones_like(points3D_normals[..., :1])), -1
            )
            @ transform_matrix.T
        )
        return {
            "points3D_normals": points3D_normals
        }
    
    # """Custom function to load touch point clouds"""
    # def load_touch_points(self, path, transform_matrix: torch.Tensor):
    #     points_from_touches = o3d.geometry.PointCloud()
    #     points_from_touches.points = o3d.utility.Vector3dVector(points.numpy())
    #     points_from_touches.colors = o3d.utility.Vector3dVector(colors.numpy())
    #     """Estimate normals to add to the touch points"""
    #     points_from_touches.estimate_normals(
    #         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    #     )
    #     points_from_touches.normalize_normals()
    #     """Apply transformation"""
    #     points_from_touches = (
    #         torch.cat(
    #             (points_from_touches, torch.ones_like(points_from_touches[..., :1])), -1
    #         )
    #         @ transform_matrix.T
    #     )
    #     return {"touch_points": points_from_touches}
    
    def _generate_dataparser_outputs(self, split="train"):
        assert (
            self.config.data.exists()
        ), f"Data directory {self.config.data} does not exist."
        self.normal_save_dir = self.config.output_dir / Path("normals_from_pretrain")
        
        meta = load_from_json(self.config.data / "transforms.json")
        data_dir = self.config.data
        output_dir = self.config.output_dir
        self.touch_meta_dir = self.config.data / "gelsight_transform.json"
        self.touch_data_dir = self.config.data / "tactile"

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2", "distortion_params"]:
            if distort_key in meta:
                distort_fixed = True
                break
        fisheye_crop_radius = meta.get("fisheye_crop_radius", None)
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        # sort the frames by fname
        fnames = []
        for frame in meta["frames"]:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)
            fnames.append(fname)
        inds = np.argsort(fnames)
        frames = [meta["frames"][ind] for ind in inds]

        for frame in frames:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    torch.tensor(frame["distortion_params"], dtype=torch.float32)
                    if "distortion_params" in frame
                    else camera_utils.get_distortion_params(
                        k1=float(frame["k1"]) if "k1" in frame else 0.0,
                        k2=float(frame["k2"]) if "k2" in frame else 0.0,
                        k3=float(frame["k3"]) if "k3" in frame else 0.0,
                        k4=float(frame["k4"]) if "k4" in frame else 0.0,
                        p1=float(frame["p1"]) if "p1" in frame else 0.0,
                        p2=float(frame["p2"]) if "p2" in frame else 0.0,
                    )
                )

            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            if "mask_path" in frame:
                mask_filepath = Path(frame["mask_path"])
                mask_fname = self._get_fname(
                    mask_filepath,
                    output_dir,
                    downsample_folder_prefix="masks_",
                )
                mask_filenames.append(mask_fname)

            if "depth_file_path" in frame:
                depth_filepath = Path(frame["depth_file_path"])
                depth_fname = self._get_fname(
                    depth_filepath, data_dir, downsample_folder_prefix="depths_"
                )
                depth_filenames.append(depth_fname)

        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (
            len(depth_filenames) == len(image_filenames)
        ), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """

        # set files in natsorted order
        normal_filenames = self.get_normal_filepaths() 
        sorted_filenames = natsorted(image_filenames)
        indices = [image_filenames.index(path) for path in sorted_filenames]
        image_filenames = [image_filenames[i] for i in indices]
        depth_filenames = [depth_filenames[i] for i in indices]
        poses = [poses[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if mask_filenames else []

        has_split_files_spec = any(
            f"{split}_filenames" in meta for split in ("train", "val", "test")
        )
        if f"{split}_filenames" in meta:
            # Validate split first
            split_filenames = set(
                self._get_fname(Path(x), data_dir) for x in meta[f"{split}_filenames"]
            )
            unmatched_filenames = split_filenames.difference(image_filenames)
            if unmatched_filenames:
                raise RuntimeError(
                    f"Some filenames for split {split} were not found: {unmatched_filenames}."
                )

            indices = [
                i for i, path in enumerate(image_filenames) if path in split_filenames
            ]
            CONSOLE.log(f"[yellow] Dataset is overriding {split}_indices to {indices}")
            indices = np.array(indices, dtype=np.int32)
        # elif has_split_files_spec:
        #     raise RuntimeError(
        #         f"The dataset's list of filenames for split {split} is missing."
        #     )
        # else:
        #     # find train and eval indices based on the eval_mode specified
        #     if self.config.eval_mode == "fraction":
        #         i_train, i_eval = get_train_eval_split_fraction(
        #             image_filenames, self.config.train_split_fraction
        #         )
        #     elif self.config.eval_mode == "filename":
        #         i_train, i_eval = get_train_eval_split_filename(image_filenames)
        #     elif self.config.eval_mode == "interval":
        #         i_train, i_eval = get_train_eval_split_interval(
        #             image_filenames, self.config.eval_interval
        #         )
        #     elif self.config.eval_mode == "all":
        #         CONSOLE.log(
        #             "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
        #         )
        #         i_train, i_eval = get_train_eval_split_all(image_filenames)
        #     else:
        #         raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

        #     if split == "train":
        #         indices = i_train
        #     elif split in ["val", "test"]:
        #         indices = i_eval
        #     else:
        #         raise ValueError(f"Unknown dataparser split {split}")

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(
                f"[yellow] Dataset is overriding orientation method to {orientation_method}"
            )
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses[:, :3, 1:3] *= -1     # FusionSense to nerfstudio format, quote this out for Touch-GS
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        # image_filenames = natsorted(image_filenames)
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = (
            [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        )
        depth_filenames = (
            [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []
        )
        normal_filenames = (
            [Path(normal_filenames[i]) for i in indices]
            if len(normal_filenames) > 0
            else []
        )

        stems = [name.stem for name in image_filenames]
        for name in normal_filenames:
            assert name.stem in stems, name

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [
                    [-aabb_scale, -aabb_scale, -aabb_scale],
                    [aabb_scale, aabb_scale, aabb_scale],
                ],
                dtype=torch.float32,
            )
        )
        # scene_box = SceneBox(
        #     aabb=torch.tensor(
        #         [
        #             [-aabb_scale, -aabb_scale, -aabb_scale],
        #             [aabb_scale, aabb_scale, aabb_scale],
        #         ],
        #         dtype=torch.float32,
        #     )
        # )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        fx = (
            float(meta["fl_x"])
            if fx_fixed
            else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        )
        fy = (
            float(meta["fl_y"])
            if fy_fixed
            else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        )
        cx = (
            float(meta["cx"])
            if cx_fixed
            else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        )
        cy = (
            float(meta["cy"])
            if cy_fixed
            else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        )
        height = (
            int(meta["h"])
            if height_fixed
            else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        )
        width = (
            int(meta["w"])
            if width_fixed
            else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        )
        if distort_fixed:
            distortion_params = (
                torch.tensor(meta["distortion_params"], dtype=torch.float32)
                if "distortion_params" in meta
                else camera_utils.get_distortion_params(
                    k1=float(meta["k1"]) if "k1" in meta else 0.0,
                    k2=float(meta["k2"]) if "k2" in meta else 0.0,
                    k3=float(meta["k3"]) if "k3" in meta else 0.0,
                    k4=float(meta["k4"]) if "k4" in meta else 0.0,
                    p1=float(meta["p1"]) if "p1" in meta else 0.0,
                    p2=float(meta["p2"]) if "p2" in meta else 0.0,
                )
            )
        else:
            distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        # Only add fisheye crop radius parameter if the images are actually fisheye, to allow the same config to be used
        # for both fisheye and non-fisheye datasets.
        metadata = {}
        if (camera_type in [CameraType.FISHEYE, CameraType.FISHEYE624]) and (
            fisheye_crop_radius is not None
        ):
            metadata["fisheye_crop_radius"] = fisheye_crop_radius

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            metadata=metadata,
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        # The naming is somewhat confusing, but:
        # - transform_matrix contains the transformation to dataparser output coordinates from saved coordinates.
        # - dataparser_transform_matrix contains the transformation to dataparser output coordinates from original data coordinates.
        # - applied_transform contains the transformation to saved coordinates from original data coordinates.
        applied_transform = None
        colmap_path = self.config.data / "colmap/sparse/0"
        if "applied_transform" in meta:
            applied_transform = torch.tensor(
                meta["applied_transform"], dtype=transform_matrix.dtype
            )
        elif colmap_path.exists():
            # For converting from colmap, this was the effective value of applied_transform that was being
            # used before we added the applied_transform field to the output dataformat.
            meta["applied_transform"] = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0]]
            applied_transform = torch.tensor(
                meta["applied_transform"], dtype=transform_matrix.dtype
            )

        if applied_transform is not None:
            dataparser_transform_matrix = transform_matrix @ torch.cat(
                [
                    applied_transform,
                    torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype),
                ],
                0,
            )
        else:
            dataparser_transform_matrix = transform_matrix

        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        # reinitialize metadata for dataparser_outputs
        metadata = {}

        # _generate_dataparser_outputs might be called more than once so we check if we already loaded the point cloud
        try:
            self.prompted_user
        except AttributeError:
            self.prompted_user = False

        # Load 3D points
        if self.config.load_3D_points:
            if "ply_file_path" in meta:
                ply_file_path = output_dir / meta["ply_file_path"]

            elif colmap_path.exists():
                from rich.prompt import Confirm

                # check if user wants to make a point cloud from colmap points
                if not self.prompted_user:
                    self.create_pc = Confirm.ask(
                        "load_3D_points is true, but the dataset was processed with an outdated ns-process-data that didn't convert colmap points to .ply! Update the colmap dataset automatically?"
                    )

                if self.create_pc:
                    import json

                    from nerfstudio.process_data.colmap_utils import (
                        create_ply_from_colmap,
                    )

                    with open(self.config.data / "transforms.json") as f:
                        transforms = json.load(f)

                    # Update dataset if missing the applied_transform field.
                    if "applied_transform" not in transforms:
                        transforms["applied_transform"] = meta["applied_transform"]

                    if "ply_file_path" not in transforms:
                        ply_filename = "sparse_pc.ply"
                        create_ply_from_colmap(
                            filename=ply_filename,
                            recon_dir=colmap_path,
                            output_dir=self.config.output_dir,
                            applied_transform=applied_transform,
                        )
                        ply_file_path = output_dir / ply_filename
                        transforms["ply_file_path"] = ply_filename

                    # This was the applied_transform value

                    with open(
                        self.config.data / "transforms.json", "w", encoding="utf-8"
                    ) as f:
                        json.dump(transforms, f, indent=4)
                else:
                    ply_file_path = None
            else:
                if not self.prompted_user:
                    CONSOLE.print(
                        "[bold yellow]Warning: load_3D_points set to true but no point cloud found. splatfacto will use random point cloud initialization."
                    )
                ply_file_path = None

            if ply_file_path:
                sparse_points = self._load_3D_points(
                    ply_file_path, transform_matrix, scale_factor
                )
                if sparse_points is not None:
                    metadata.update(sparse_points)
            self.prompted_user = True

        if "object_pc_path" in meta:
            object_pc_path = output_dir / meta["object_pc_path"]
            visual_hull = self._load_3D_points(
                object_pc_path, transform_matrix, scale_factor
            )
            visual_hull_pts = {"visual_hull": visual_hull["points3D_xyz"]}
            if visual_hull is not None:
                metadata.update(visual_hull_pts)

            # if meta["mesh_aabb"]:
            #     xyz_min = visual_hull_pts["visual_hull"][:, :3].min(axis=0).values - 0.05
            #     xyz_max = visual_hull_pts["visual_hull"][:, :3].max(axis=0).values + 0.05
            #     print(xyz_min, xyz_max)
            #     # aabb_scale = (xyz_max - xyz_min).max() / 2 + 0.05
            #     scene_box = SceneBox(
            #         aabb=torch.tensor(
            #             [
            #                 [xyz_min[0], xyz_min[1], xyz_min[2]],
            #                 [xyz_max[0], xyz_max[1], xyz_max[2]],
            #             ],
            #             dtype=torch.float32,
            #         )
            #     )

        if self.config.load_pcd_normals:
            metadata.update(
                self._load_points3D_normals(
                    points=metadata["points3D_xyz"],
                    colors=metadata["points3D_rgb"],
                    transform_matrix=transform_matrix,
                )
            )

        if self.config.load_normals:
            metadata["normal_filenames"] = normal_filenames
            metadata["load_normals"] = True
            metadata["normal_format"] = self.config.normal_format

        if self.config.load_touches:
            metadata["load_touches"] = True
            CONSOLE.log("[bold green] Load touches...")
            touch_meta = load_from_json(self.touch_meta_dir)
            touchframes = touch_meta["frames"]
            metadata["touch_filenames"] = self.get_touch_filepaths()
            touch_patches = []
            for ind, touchframe in zip(range(len(touchframes)), touchframes):
                # read touch patch from pcd/ply file
                pts = o3d.io.read_point_cloud(str(self.config.data / touchframe["patch_path"]))
                raw_pcd = torch.from_numpy(np.asarray(pts.points)).to(dtype=torch.float32)
                touch_downsample_factor = 5
                before = raw_pcd.shape[0]
                raw_pcd = raw_pcd[:: touch_downsample_factor, :]
                # assert before == raw_pcd.shape[0]*touch_downsample_factor, f"{before}*{touch_downsample_factor} != {raw_pcd.shape[0]}"
                tr = torch.tensor(touchframe["transform_matrix"], dtype=raw_pcd.dtype)  # 4x4 homogenous transform matrix
                # centralize
                pcd = raw_pcd.clone()
                pcd[:, :2] -= torch.mean(raw_pcd, dim=0)[:2]
                # pcd[:, 0] *= -1
                # pcd[:, 1] *= -1
                # our pcd data has integer xy index, so need to multiply by the factor
                pcd *= self.config.gel_scale_factor
                pcd = mut_and_scale(pcd, tr[:3, :], 1.0)
                pcd = mut_and_scale(pcd, transform_matrix, scale_factor)
                
                # read and apply touch patch mask
                if (touchframe["mask_path"].endswith(".pcd")):
                    mask_pcd = np.asarray(o3d.io.read_point_cloud(str(self.config.data / touchframe["mask_path"])).points)
                    mask = mask_pcd[:, 2] == 1
                elif (touchframe["mask_path"].endswith(".npy")):
                    mask = np.load(str(self.config.data / touchframe["mask_path"]))
                else:
                    raise KeyError("Unsupported mask type")
                mask = mask[:: touch_downsample_factor]
                np_pts = pcd[mask]
                the_pcd = o3d.geometry.PointCloud()
                the_pcd.points = o3d.utility.Vector3dVector(np_pts.numpy())
                # o3d.io.write_point_cloud(f"validate_touch/patch_{ind}.ply", the_pcd)
                ## save pcd
                # the_pcd = o3d.geometry.PointCloud()
                # the_pcd.points = o3d.utility.Vector3dVector(np_pts.numpy())
                # o3d.io.write_point_cloud(f"patch_{ind}_new.ply", the_pcd)
                ## load normals as 3D normal vectors
                pcd_normals = torch.from_numpy(np.load(self.config.data / Path(touchframe["normal_path"])))
                pcd_normals = pcd_normals[:: touch_downsample_factor, :]
                if (pcd_normals.shape[-1]==2):
                    pcd_normals = pcd_normals.reshape(-1, 2)[mask] # MxNx2 file --> (MxN)x3
                    x = pcd_normals[..., 0]
                    y = pcd_normals[..., 1]
                    z = -np.sqrt(np.maximum(1.0 - x**2 - y**2, 0.0))
                elif (pcd_normals.shape[-1]==3):
                    pcd_normals = pcd_normals.reshape(-1, 3)[mask] # MxNx2 file --> (MxN)x3
                    x = pcd_normals[..., 0]
                    y = pcd_normals[..., 1]
                    z = pcd_normals[..., 2]
                else:
                    raise KeyError("Unsupported Normal Type")
                pcd_normal3D = torch.stack((x, y, z)).to(dtype=torch.float32).T
                pcd_normal3D = mut_and_scale(pcd_normal3D, tr[:3, :], 1.0) # apply transformation
                # non-axis aligned bbox 
                x_diff = torch.abs(torch.max(raw_pcd[:, 0]) - torch.min(raw_pcd[:, 0]))
                y_diff = torch.abs(torch.max(raw_pcd[:, 1]) - torch.min(raw_pcd[:, 1]))
                z_diff = torch.abs(torch.max(raw_pcd[:, 2]) - torch.min(raw_pcd[:, 2])) # raw pcd z coordinate is negative
                min_corner = [-x_diff/2, -y_diff/2, -z_diff*5]
                max_corner = [x_diff/2, y_diff/2, 0]
                aabb = torch.tensor([
                    [min_corner[0], min_corner[1], min_corner[2]],
                    [min_corner[0], min_corner[1], max_corner[2]],
                    [min_corner[0], max_corner[1], min_corner[2]],
                    [min_corner[0], max_corner[1], max_corner[2]],
                    [max_corner[0], min_corner[1], min_corner[2]],
                    [max_corner[0], min_corner[1], max_corner[2]],
                    [max_corner[0], max_corner[1], min_corner[2]],
                    [max_corner[0], max_corner[1], max_corner[2]],
                ], dtype=pcd.dtype)
                aabb *= self.config.gel_scale_factor
                aabb = mut_and_scale(aabb, tr[:3,:], 1.0)
                aabb = mut_and_scale(aabb, transform_matrix, scale_factor)
                # o3d.io.write_point_cloud(f"validate_touch/{ind}_aabb.ply", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(aabb)))
                
                assert pcd_normal3D.dtype == np_pts.dtype == tr.dtype
                touch_patch = {
                    "points_xyz": np_pts,
                    "points_rgb": torch.zeros_like(np_pts, dtype=pcd.dtype), # init touch points to be black
                    "normals": pcd_normal3D,
                    "bbox": aabb,
                }
                touch_patches.append(touch_patch)
                ## add to array of patches
            metadata.update({"touch_patches": touch_patches})
            metadata.update({"gel_scale_factor": self.config.gel_scale_factor})
            # CONSOLE.log("[bold red] Warning: no touch patches were loaded, check your touch file path and params")

        scale_factor_dict = {"scale_factor": scale_factor}
        metadata.update(scale_factor_dict)

        transform_matrix_dict = {"transform_matrix": transform_matrix}
        metadata.update(transform_matrix_dict)

        grad_visualization_dict = {'grad_visualization': self.config.grad_visualization}
        metadata.update(grad_visualization_dict)

        if self.config.load_cameras:
            import json
            with open(self.config.camera_path_filename, "r", encoding="utf-8") as f:
                camera_path = json.load(f)
            camera_path = get_path_from_json(camera_path)
            laod_cameras_dict = {"load_cameras": self.config.load_cameras}
            camera_path_dict = {"camera_path": camera_path}
            metadata.update(laod_cameras_dict)
            metadata.update(camera_path_dict)
        else:
            laod_cameras_dict = {"load_cameras": self.config.load_cameras}
            metadata.update(laod_cameras_dict)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=dataparser_transform_matrix,
            metadata={
                "depth_filenames": depth_filenames
                if len(depth_filenames) > 0
                else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                "mask_color": self.config.mask_color,
                **metadata,
            },
        )
        return dataparser_outputs


NormalNerfstudioSpecification = DataParserSpecification(
    config=NormalNerfstudioConfig(),
    description="Nerfstudio dataparser that loads normals",
)

if __name__ == "__main__":
    parser = NormalNerfstudio(NormalNerfstudioConfig)._generate_dataparser_outputs()
