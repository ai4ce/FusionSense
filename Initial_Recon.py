import os
import json
import subprocess
from dataclasses import dataclass
from utils.imgs_selection import select_imgs, filter_transform_json
from utils.VisualHull import VisualHull
from utils.metric3dv2_depth_generation import metric3d_depth_generation
from utils.generate_pcd import Init_pcd_generate

@dataclass
class GSReconstructionConfig:
    steps_per_save: int = 15000
    iterations: int = 15001

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

    model_type: str = "normal-nerfstudio"

class Initial_Reconstruction:
    def __init__(self, base_path):
        # 初始化通用路径
        self.base_path = base_path
        with open(os.path.join(base_path, 'transforms.json'), 'r') as f:
            self.transforms = json.load(f)
    
    def select_frames(self):
        select_imgs(self.base_path)
        filter_transform_json(self.base_path)
    
    def generate_mask_images(self, absolute_path, prompt_text):
        """Step 2: Generate Mask Images using Grounded SAM"""
        print("Generating mask images...")
        os.chdir(self.grounded_sam_path)  # 切换到 Grounded-SAM 目录
        command = f"python grounded_sam2_hf_model_imgs_MaskExtract.py --path {absolute_path} --prompt '{prompt_text}'"
        subprocess.run(command, shell=True, check=True)
        print("Mask images generated.")
    
    def generate_visual_hull(self, error):
        VisualHull(self.base_path, error)
    
    def run_metric3d_depth(self):
        fl_x = self.transforms['fl_x']
        fl_y = self.transforms['fl_y']
        cx = self.transforms['cx']
        cy = self.transforms['cy']
        H = self.transforms['h']
        W = self.transforms['w']
        img_dir = self.transforms['frames'][0]['file_path'].split('/')[0]
        intrinsics = [fl_x, 0, cx, 0, fl_y, cy, 0, 0, 1]
        frame_size = [W, H]
        metric3d_depth_generation(self.base_path, intrinsics, frame_size, img_dir=img_dir)
    
    def Init_pcd_generation(self):
        Init_pcd_generate(self.base_path)
    
    def generate_normals(self):
        """Step 6: Generate normals"""
        print("Generating normals...")
        command = f"python dn_splatter/scripts/normals_from_pretrain.py --data-dir {self.base_path} --model-type dsine"
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
    
    def train_model(self):
        configs = GSReconstructionConfig(data_path=self.base_path)
        if configs.data_path == None:
            assert False, "Please set data_path in GSReconstructionConfig"
        command = [
            "ns-train",
            "dn-splatter",
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
            str(configs.model_type),
            "--data", configs.data_path,
            "--load-pcd-normals", str(configs.load_pcd_normals),
            "--load-3D-points", str(configs.load_3D_points),
            "--normal-format", configs.normal_format,
            "--load-touches", str(configs.load_touches)
        ]

        # command = "CUDA_VISIBLE_DEVICES=0 ns-train dn-splatter --steps-per-save 30000 --max_num_iterations 30001 --pipeline.model.use-depth-loss True --pipeline.model.normal-lambda 0.4 --pipeline.model.sensor-depth-lambda 0.2 --pipeline.model.use-depth-smooth-loss True  --pipeline.model.use-binary-opacities True  --pipeline.model.use-normal-loss True  --pipeline.model.normal-supervision mono  --pipeline.model.random_init False normal-nerfstudio  --data datasets/touchgs  --load-pcd-normals True --load-3D-points True  --normal-format opencv"
        print(command)
        print("Training the model...")
        subprocess.run(command)
        print("Training complete.")
    
    def extract_mesh(self, config_path, output_dir):
        """Step 9: Extract mesh"""
        print("Extracting mesh...")
        command = f"gs-mesh dn --load-config {config_path} --output-dir {output_dir}"
        subprocess.run(command, shell=True, check=True)
        print("Mesh extracted.")
    
    def export_gsplats(self, config_path, output_dir):
        """Step 10: Export GSplat"""
        print("Exporting GSplat...")
        command = f"ns-export gaussian-splat --load-config {config_path} --output-dir {output_dir}"
        subprocess.run(command, shell=True, check=True)
        print("GSplat exported.")

# 示例用法
if __name__ == "__main__":
    init_recon = Initial_Reconstruction(base_path="datasets/blackbunny3")

    init_recon.select_frames()
    # init_recon.generate_mask_images(absolute_path="/absolute/path/to/your/data", prompt_text="transparent white statue.")
    init_recon.generate_visual_hull(error=5)
    # init_recon.run_metric3d_depth()
    init_recon.Init_pcd_generation()
    init_recon.generate_normals()
    init_recon.set_transforms_and_configs()
    
    # train the model
    init_recon.train_model()

    # init_recon.extract_mesh(config_path="path/to/config.yml", output_dir="path/to/output")
    # init_recon.export_gsplats(config_path="outputs/unnamed/dn-splatter/2024-09-02_203650/config.yml", output_dir="exports/splat/")
