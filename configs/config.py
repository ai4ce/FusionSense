from pathlib import Path

steps_per_save: int = 15000
iterations: int = 15000
stop_split_at: int = 10000
warmup_length: int = 500
add_touch_at: int = 1000

use_depth_loss: bool = True
normal_lambda: float = 0.4
sensor_depth_lambda: float = 0.2
use_depth_smooth_loss: bool = True
use_binary_opacities: bool = True
use_normal_loss: bool = True
normal_supervision: str = "mono"
load_pcd_normals: bool = True
load_3D_points: bool = True
load_touches: bool = False
load_cameras: bool = False
camera_path_filename: Path = Path("camera_path.json")

config = dict(
    steps_per_save=steps_per_save,
    iterations=iterations,
    stop_split_at=stop_split_at,
    warmup_length=warmup_length,
    add_touch_at=add_touch_at,
    use_depth_loss=use_depth_loss,
    normal_lambda=normal_lambda,
    sensor_depth_lambda=sensor_depth_lambda,
    use_depth_smooth_loss=use_depth_smooth_loss,
    use_binary_opacities=use_binary_opacities,
    use_normal_loss=use_normal_loss,
    normal_supervision=normal_supervision,
    load_pcd_normals=load_pcd_normals,
    load_3D_points=load_3D_points,
    load_touches=load_touches,
    load_cameras=load_cameras,
    camera_path_filename=camera_path_filename,
)