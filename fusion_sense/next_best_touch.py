import os
import torch
from pytorch3d.io import IO
import trimesh
import numpy as np
from pathlib import Path
import glob

import base64
from openai import OpenAI
from pydantic import BaseModel

from typing import Union

from partslip.partslip_src.utils import normalize_pc
from partslip.partslip_src.render_pc import render_pc
from partslip.partslip_src.glip_inference import glip_inference, load_model
from partslip.partslip_src.gen_superpoint import gen_superpoint
from partslip.partslip_src.bbox2seg import bbox2seg

from ament_index_python.packages import get_package_share_directory

class PartResponse(BaseModel):
    '''
    This class define the struction of the response from OpenAI API
    '''
    classification: str
    parts: list[str]

class NextBestTouch:
    '''
    This class is responsible for the next best touch prediction
    '''
    def __init__(self, folder_path):
        '''
        Args:
            folder_path: str
                Path to the fusion_sense_resource folder
        '''
        self.folder_path = folder_path
        self.partslip_folder = os.path.join(self.folder_path, "partslip")
        api_path = os.path.join(self.folder_path, 'api_key.txt')
        with open(api_path, 'r') as file:
            api_key = file.read()

        self.client = OpenAI(api_key=api_key)

    def next_best_touch_prediction(self):
        image_folder_path = os.path.join(self.folder_path, "images")
        images = glob.glob(os.path.join(image_folder_path, "*.png"))

        mesh_path = os.path.join(self.partslip_folder, "meshes")
        mesh = glob.glob(os.path.join(mesh_path, "*.obj"))[0]
        point_cloud_path = self.pointcloud_extraction(mesh)

        for image in images:
            classification, parts = self.partname_extraction(image)
            print(f'[2. The object is classified as: {classification}, and the parts are: {parts}]')
            if classification is None:
                continue
            self.partslip_infer(point_cloud_path, parts, save_dir=os.path.join(self.partslip_folder, 'output'))
            break
        

    def pointcloud_extraction(self, path_to_mesh, num_points=100000):
        '''
        Downsample the mesh from step one to point cloud

        Args:
            path_to_mesh: str or Path
                Path to the mesh file
            num_points: int
                Number of points to sample from the mesh
        '''

        print('[2. Generating dense point cloud from the mesh...]')
        mesh_name = Path(path_to_mesh).stem
        self.object_name = mesh_name
        mesh = trimesh.load(path_to_mesh, process=False)


        # Sample points from the surface of the mesh
        points, face_indices = trimesh.sample.sample_surface(mesh, num_points)

        # If the mesh has vertex colors, we need to interpolate them for the sampled points
        if hasattr(mesh.visual, 'vertex_colors'):
            # Extract face colors
            face_colors = mesh.visual.face_colors[:, :3] / 255.0  # Normalize RGB
            # Get the colors corresponding to the sampled points
            sampled_colors = face_colors[face_indices]
        else:
            # Default to black if no colors are found
            sampled_colors = np.ones_like(points) * [0, 0, 0]

        # Save the denser point cloud to a .ply file
        output_ply_file = os.path.join(self.partslip_folder, 'pointcloud', Path(f'{self.object_name}.ply'))
        point_cloud = trimesh.points.PointCloud(points, sampled_colors)
        point_cloud.export(output_ply_file)

        print(f"Dense point cloud saved to {output_ply_file}")
        return output_ply_file

    def partname_extraction(self, image_path):

        print('[2. Querying VLM model for part names...]')

        # Path to your image
        image_path = "/home/irving/Desktop/tactile_ws/src/PartSLIP/unnamed.png"

        # Getting the base64 string
        base64_image = self._encode_image(image_path)

        PROMPT_MESSAGES = [
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "Take a deep breath. You will be given a picture with an object in the center. First, tell the classification of the object. Then, You will need to describe the major parts that make up the object. Use label-like everyday single words. When giving the label, think of parts that are difficult to have a good perception purly based on vision, but would also rely on tactile perception. For example a small button on a earphone case. Separate your answer with commas and end with a period."
                }
            ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    },
                ],
            },
        ]
        
        response = self._call_openai_api(PROMPT_MESSAGES)

        if response is None:
            print("Failed to get a response from OpenAI API")
            return None, None
        
        return response.classification, response.parts

    def partslip_infer(self, input_pc_file, part_names, save_dir="tmp"):
        config ="GLIP/configs/glip_Swin_L.yaml"
        weight_path = "models/glip_large_model.pth"
        
        print("[2. Loading GLIP model...]")
        glip_demo = load_model(config, weight_path)

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        io = IO()
        os.makedirs(save_dir, exist_ok=True)
        
        print("[2. Normalizing input point cloud...]")
        xyz, rgb = normalize_pc(input_pc_file, save_dir, io, device)
        
        print("[2. Rendering input point cloud...]")
        img_dir, pc_idx, screen_coords = render_pc(xyz, rgb, save_dir, device)
        
        print("[2. Glip infrence...]")
        preds = glip_inference(glip_demo, save_dir, part_names)
        
        print('[2. Generating superpoints...]')
        superpoint = gen_superpoint(xyz, rgb, visualize=True, save_dir=save_dir)
        
        print('[2. Generating 3D part segmentation...]')
        sem_seg, ins_seg = bbox2seg(xyz, superpoint, preds, screen_coords, pc_idx, part_names, save_dir, solve_instance_seg=True)
    

    def _encode_image(self, image_path):
        '''
        Encode the image to base64 string for OpenAI API
        '''
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _call_openai_api(self, prompt_messages) -> Union[PartResponse, None]:
        '''
        Wrapper function to call OpenAI API
        '''
        params = {
            "model": "gpt-4o-2024-08-06",
            "messages": prompt_messages,
            "max_tokens": 400,
            "temperature": 0,
            "response_format": PartResponse,
        }
        result = self.client.beta.chat.completions.parse(**params)
        return result.choices[0].message.parsed


def main():
    fusion_sense_folder = os.path.join(get_package_share_directory('fusion_sense'), 'fusion_sense_resources')
    nbt = NextBestTouch(fusion_sense_folder)
    nbt.next_best_touch_prediction()
if __name__ == "__main__":
    main()