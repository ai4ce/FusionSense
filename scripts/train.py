import os
import sys
sys.path.insert(0, os.getcwd())
import json
import torch
import subprocess
import argparse
import contextlib
from io import StringIO
from pathlib import Path

from dataclasses import dataclass
from utils.imgs_selection import select_imgs, filter_transform_json
from utils.VisualHull import VisualHull
from utils.metric3dv2_depth_generation import metric3d_depth_generation
from utils.generate_pcd import init_pcd_generate
from eval_utils.rendering_evaluation import rendering_evaluation
from eval_utils.chamfer_evaluation import chamfer_eval
from eval_utils.mask_rendering_eval import mask_rendering_evaluation
from nerfstudio.utils.rich_utils import CONSOLE
from importlib.machinery import SourceFileLoader

@contextlib.contextmanager
def suppress_output(verbose=False):
    # If verbose is False, suppress output
    if not verbose:
        original_stdout = sys.stdout
        sys.stdout = StringIO()  # Redirect stdout to an in-memory object
    try:
        yield
    finally:
        if not verbose:
            sys.stdout = original_stdout  # Restore stdout if it was suppressed

@dataclass
class GSReconstructionConfig:
    output_dir: Path = Path("outputs")
    steps_per_save: int = 15000
    iterations: int = 15000
    stop_split_at: int = 10000

    use_depth_loss: bool = True
    normal_lambda: float = 0.4
    sensor_depth_lambda: float = 0.2
    use_depth_smooth_loss: bool = True
    use_binary_opacities: bool = True
    use_normal_loss: bool = True
    normal_supervision: str = "mono"
    random_init: bool = False

    data_path: str = None
    load_pcd_normals: bool = True
    load_3D_points: bool = True
    normal_format: str = "opencv"
    load_touches: bool = False
    load_cameras: bool = False
    camera_path_filename: Path = Path("camera_path.json")

    model_type: str = "normal-nerfstudio"
    warmup_length: int = 500
    add_touch_at: int = 1000

class Initial_Reconstruction:
    def __init__(self, data_name, model_name, prompt_text='Near Object'):
        self.data_name = data_name
        self.model_name = model_name
        self.base_path = os.path.join("datasets", self.data_name)
        self.output_dir = os.path.join("outputs", self.data_name, self.model_name)
        self.eval_dir = os.path.join("eval", self.data_name, self.model_name)
        # self.prompt_text = prompt_text
        self.grounded_sam_path = "Grounded-SAM2-for-masking"
        with open(os.path.join(self.base_path, 'transforms.json'), 'r') as f:
            self.transforms = json.load(f)
    
    def select_frames(self):
        select_imgs(self.base_path, self.output_dir)
        filter_transform_json(self.base_path)
    
    def generate_visual_hull(self, error):
        VisualHull(self.base_path, self.output_dir, error)
    
    def run_metric3d_depth(self, vram_size="large"):
        fl_x = self.transforms['fl_x']
        fl_y = self.transforms['fl_y']
        cx = self.transforms['cx']
        cy = self.transforms['cy']
        H = self.transforms['h']
        W = self.transforms['w']
        img_dir = self.transforms['frames'][0]['file_path'].split('/')[0]
        intrinsics = [fl_x, 0, cx, 0, fl_y, cy, 0, 0, 1]
        frame_size = [W, H]
        metric3d_depth_generation(self.base_path, self.output_dir, intrinsics, frame_size, img_dir=img_dir, vram_size=vram_size)
    
    def init_pcd_generation(self):
        init_pcd_generate(self.base_path, self.output_dir)
    
    def generate_normals(self):
        """Step 6: Generate normals"""
        print("Generating normals...")
        save_path = Path(self.output_dir) / "normals_from_pretrain"
        command = f"python dn_splatter/scripts/normals_from_pretrain.py --data-dir {self.base_path} --save-path {save_path} --model-type dsine"
        subprocess.run(command, shell=True, check=True)
        print("Normals generated.")
    
    def set_transforms_and_configs(self):
        with open(os.path.join(self.base_path, 'transforms.json'), 'r') as f:
            transforms = json.load(f)
        transforms['ply_file_path'] = "merged_pcd.ply"
        transforms['object_pc_path'] = "foreground_pcd.ply"
        transforms["mesh_aabb"] = True
        with open(os.path.join(self.base_path, 'transforms.json'), 'w') as f:
            json.dump(transforms, f, indent=4)
    
    def train_model(self, configs: GSReconstructionConfig):
        if configs.data_path == None:
            assert False, "Please set data_path in GSReconstructionConfig"
        command = [
            "ns-train",
            "dn-splatter",
            "--output-dir", str(configs.output_dir),
            "--steps-per-save", str(configs.steps_per_save),
            "--max_num_iterations", str(configs.iterations),
            "--pipeline.model.use-depth-loss", str(configs.use_depth_loss),
            "--pipeline.model.normal-lambda", str(configs.normal_lambda),
            "--pipeline.model.sensor-depth-lambda", str(configs.sensor_depth_lambda),
            "--pipeline.model.use-depth-smooth-loss", str(configs.use_depth_smooth_loss),
            "--pipeline.model.use-binary-opacities", str(configs.use_binary_opacities),
            "--pipeline.model.use-normal-loss", str(configs.use_normal_loss),
            "--pipeline.model.normal-supervision", configs.normal_supervision,
            "--pipeline.model.random_init", str(configs.random_init),
            "--pipeline.model.warmup-length", str(configs.warmup_length),
            "--pipeline.model.add-touch-at", str(configs.add_touch_at),
            "--pipeline.model.stop-split-at", str(configs.stop_split_at),
            "--pipeline.model.base-dir", str(configs.output_dir),
            "--viewer.quit-on-train-completion", 'True',
            str(configs.model_type),
            "--data", configs.data_path,
            "--output-dir", configs.output_dir,
            "--load-pcd-normals", str(configs.load_pcd_normals),
            "--load-3D-points", str(configs.load_3D_points),
            "--normal-format", configs.normal_format,
            "--load-touches", str(configs.load_touches),
            "--load-cameras", str(configs.load_cameras),
            "--camera-path-filename", configs.camera_path_filename,
        ]

        # command = "CUDA_VISIBLE_DEVICES=0 ns-train dn-splatter --steps-per-save 30000 --max_num_iterations 30001 --pipeline.model.use-depth-loss True --pipeline.model.normal-lambda 0.4 --pipeline.model.sensor-depth-lambda 0.2 --pipeline.model.use-depth-smooth-loss True  --pipeline.model.use-binary-opacities True  --pipeline.model.use-normal-loss True  --pipeline.model.normal-supervision mono  --pipeline.model.random_init False normal-nerfstudio  --data datasets/touchgs  --load-pcd-normals True --load-3D-points True  --normal-format opencv"
        print(command)
        print("Training the model...")
        subprocess.run(command)
        print("Training complete.")

    def add_touch_train_model(self, configs: GSReconstructionConfig):
        configs.load_touches = True
        configs.output_dir = os.path.join(self.base_path, "outputs_with_touches")
        if configs.data_path == None:
            assert False, "Please set data_path in GSReconstructionConfig"
        command = [
            "ns-train",
            "dn-splatter",
            "--output-dir", str(configs.output_dir),
            "--steps-per-save", str(configs.steps_per_save),
            "--max_num_iterations", str(configs.iterations),
            "--pipeline.model.use-depth-loss", str(configs.use_depth_loss),
            "--pipeline.model.normal-lambda", str(configs.normal_lambda),
            "--pipeline.model.sensor-depth-lambda", str(configs.sensor_depth_lambda),
            "--pipeline.model.use-depth-smooth-loss", str(configs.use_depth_smooth_loss),
            "--pipeline.model.use-binary-opacities", str(configs.use_binary_opacities),
            "--pipeline.model.use-normal-loss", str(configs.use_normal_loss),
            "--pipeline.model.normal-supervision", configs.normal_supervision,
            "--pipeline.model.random_init", str(configs.random_init),
            "--pipeline.model.warmup-length", str(configs.warmup_length),
            "--pipeline.model.add-touch-at", str(configs.add_touch_at),
            "--pipeline.model.stop-split-at", str(configs.stop_split_at),
            "--pipeline.model.base-dir", str(configs.output_dir),
            str(configs.model_type),
            "--data", configs.data_path,
            "--load-pcd-normals", str(configs.load_pcd_normals),
            "--load-3D-points", str(configs.load_3D_points),
            "--normal-format", configs.normal_format,
            "--load-touches", str(configs.load_touches),
        ]
        print(command)
        print("Training the model...")
        subprocess.run(command)
        print("Training complete.")

    def extract_mesh(self, config_path):
        save_dir = os.path.join(self.output_dir, "MESH")
        command = [
            "gs-mesh",
            "tsdf",
            "--load-config", str(config_path),
            "--output-dir", str(save_dir+"/tsdf"),
        ]
        command_gs = [
            "gs-mesh",
            "gaussians",
            "--load-config", str(config_path),
            "--output-dir", str(save_dir+"/gaussian"),
        ]
        command_sugar = [
            "gs-mesh",
            "sugar-coarse",
            "--load-config", str(config_path),
            # "--output-dir", str(save_dir+"/sugar-coarse"),
            "--output-dir", str(save_dir),
        ]
        print("Extracting mesh...")
        CONSOLE.log(command_sugar)
        subprocess.run(command_gs)
        subprocess.run(command_sugar)
        print("Mesh extracted")
    
    def export_gsplats(self, config_path, output_dir):
        """Step 10: Export GSplat"""
        print("Exporting GSplat...")
        command = f"ns-export gaussian-splat --load-config {config_path} --output-dir {output_dir}"
        subprocess.run(command, shell=True, check=True)
        print("GSplat exported.")

    def evaluation(self, rendering_eval=True, mask_rendering=True, chamfer=True):
        if rendering_eval:
            rendering_evaluation(self.output_dir, self.eval_dir, self.data_name)
        if mask_rendering:
            mask_rendering_evaluation(self.base_path, self.eval_dir)
        if chamfer:
            mesh_dir = os.path.join(self.output_dir, "MESH")
            chamfer_eval(self.base_path, mesh_dir)
        print("Evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="transparent_bunny", help="Name of the dataset folder")
    parser.add_argument("--model_name", type=str, default="9view", help="Name of the model. It will impact the output and eval folder name. You can technically name this whatever you want.")
    parser.add_argument("--load_touches", type=bool, default=False, help="Whether to load tactile data")
    parser.add_argument("--configs", type=str, default="configs/config.py", help="Path to the Nerfstudio config file")
    parser.add_argument("--verbose", type=bool, default=False, help="False: Only show important logs. True: Show all logs.")
    parser.add_argument("--vram_size", type=str, default="large", help="large or small. Decides the foundation models variants used in the pipeline")
    args = parser.parse_args()

    data_name = args.data_name
    model_name = args.model_name
    load_touches_set = args.load_touches
    verbose = args.verbose
    vram_size = args.vram_size

    experiment = SourceFileLoader(os.sys.path[0], "configs/config.py").load_module()
    experiment_configs = experiment.config

    init_recon = Initial_Reconstruction(data_name, model_name)
    configs = GSReconstructionConfig(
        output_dir=init_recon.output_dir,
        data_path=init_recon.base_path,
        steps_per_save=experiment_configs["steps_per_save"],
        iterations=experiment_configs["iterations"],
        use_depth_loss=experiment_configs["use_depth_loss"],
        normal_lambda=experiment_configs["normal_lambda"],
        sensor_depth_lambda=experiment_configs["sensor_depth_lambda"],
        use_depth_smooth_loss=experiment_configs["use_depth_smooth_loss"],
        use_binary_opacities=experiment_configs["use_binary_opacities"],
        use_normal_loss=experiment_configs["use_normal_loss"],
        normal_supervision=experiment_configs["normal_supervision"],
        warmup_length=experiment_configs["warmup_length"],
        add_touch_at=experiment_configs["add_touch_at"],
        stop_split_at=experiment_configs["stop_split_at"],
        load_pcd_normals=experiment_configs["load_pcd_normals"],
        load_3D_points=experiment_configs["load_3D_points"],
        load_touches=experiment_configs["load_touches"],
        load_cameras=experiment_configs["load_cameras"],
        camera_path_filename=experiment_configs["camera_path_filename"]
    )

    if not load_touches_set:
        CONSOLE.log("[Module 1] 1/9 Selecte images for training...")
    else:
        CONSOLE.log("[Module 3] 1/4 Prepare re-training with tactile data...")
    with suppress_output(verbose):
        init_recon.select_frames()

    if not load_touches_set:
        CONSOLE.log("[Module 1] 2/9 Generate visual hull...")
    with suppress_output(verbose):
        init_recon.generate_visual_hull(error=5)
    
    if not load_touches_set:
        CONSOLE.log("[Module 1] 3/9 Running metric3d depth...")
    with suppress_output(verbose):
        init_recon.run_metric3d_depth(vram_size=vram_size)
    
    if not load_touches_set:
        CONSOLE.log("[Module 1] 4/9 Initialize pcd...")
    with suppress_output(verbose):
        init_recon.init_pcd_generation()
    
    if not load_touches_set:
        CONSOLE.log("[Module 1] 5/9 Generate normals...")
    with suppress_output(verbose):
        init_recon.generate_normals()
    
    if not load_touches_set:
        CONSOLE.log("[Module 1] 6/9 Set transforms.json...")
    with suppress_output(verbose):
        init_recon.set_transforms_and_configs()

    configs.load_touches = load_touches_set

    if not load_touches_set:
        CONSOLE.log("[Module 1] 7/9 Initialize training...")
    else:
        CONSOLE.log("[Module 3] 2/4 Add tactile data...")
    init_recon.train_model(configs=configs)
    torch.cuda.empty_cache() # Clear GPU memory so we can do extraction.

    if not load_touches_set:
        CONSOLE.log("[Module 1] 8/9 Extract mesh...")
    else:
        CONSOLE.log("[Module 3] 3/4 Extract mesh...")
    with suppress_output(verbose):
        init_recon.extract_mesh(config_path=os.path.join(configs.output_dir, "config.yml"))

    if not load_touches_set:
        CONSOLE.log("[Module 1] 9/9 Evaluate rendering...")
    else:
        CONSOLE.log("[Module 3] 4/4 Evaluate rendering...")
    with suppress_output(verbose):
        init_recon.evaluation(rendering_eval=True, mask_rendering=True, chamfer=False)