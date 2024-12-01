
import os
import sys
sys.path.insert(0, os.getcwd())

from pathlib import Path
import glob
import warnings
import argparse
import contextlib
from io import StringIO
from typing import Union, List

import torch
import numpy as np

from pytorch3d.io import IO
import trimesh
import open3d as o3d
import open3d.core as o3c

import base64
from openai import OpenAI
from pydantic import BaseModel



from PartSlip.src.utils import normalize_pc
from PartSlip.src.render_pc import render_pc
from PartSlip.src.glip_inference import glip_inference, load_model
from PartSlip.src.gen_superpoint import gen_superpoint
from PartSlip.src.bbox2seg import bbox2seg

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

warnings.filterwarnings("ignore")

class PartResponse(BaseModel):
    '''
    This class define the struction of the response from OpenAI API
    '''
    classification: str
    parts: List[str]

class VLM:
    '''
    This class is responsible for the next best touch prediction
    '''
    def __init__(self, img_folder_path):
        '''
        Args:
            output_folder: str
                Path to the output folder
        '''
        self.image_folder = img_folder_path # holds all the images

        self.OAIclient = OpenAI()

    def update_output_folder(self, output_folder_path):
        self.output_folder = output_folder_path 

        self.segmentation_folder = os.path.join(self.output_folder, "segmentation") # folder to hold all segmentation resource and results
        
    def touch_selection(self, mesh_path, object_name=None, part_name=None):
        
        self.point_cloud_path = self.pointcloud_extraction(mesh_path)

        if object_name is not None and part_name is not None:
            classification = object_name
            parts = part_name
        else:
            classification, parts = self.partname_extraction()

        self._create_rank_dict(parts) # Create a dictionary to rank the parts in terms of which one to touch first

        self.seg_output_dir = os.path.join(self.segmentation_folder, 'output', self.object_name)
        self.segmentation_infer(self.point_cloud_path, parts, save_dir=self.seg_output_dir)
        self.grounding_segmentation()
        self.fuse_gaussian_and_segmentation()
        self.propose_next_best_touch()

        
    def pointcloud_extraction(self, path_to_mesh, num_points=100000):
        '''
        Downsample the mesh from step one to point cloud

        Args:
            path_to_mesh: str or Path
                Path to the mesh file
            num_points: int
                Number of points to sample from the mesh
        '''

        print('[Module 2] 1/11 Generating dense point cloud from the mesh...')
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
        self.pointcloud_folder = os.path.join(self.segmentation_folder, 'pointcloud')
        os.makedirs(self.pointcloud_folder, exist_ok=True)
        output_ply_file = os.path.join(self.pointcloud_folder, Path(f'{self.object_name}.ply'))
        point_cloud = trimesh.points.PointCloud(points, sampled_colors)
        point_cloud.export(output_ply_file)

        return output_ply_file

    def partname_extraction(self, model_name="gpt-4o", mode="partname"):
        if mode == "partname":
            print('Getting part names with VLM...')
        elif mode == "touch":
            print('[Module 2] 2/11 Getting part names...')
        images = glob.glob(os.path.join(self.image_folder, "*"))
        for image in images:
            classification, parts = self._partname_extraction_call(image, model_name)

            if classification is None:
                continue
            else:
                print(f'The object is: {classification}, and the parts are: {parts}]')
                break
        return classification, parts

    def segmentation_infer(self, input_pc_file, part_names, save_dir="tmp"):
        '''
        Segment the point cloud into parts using the PartSlip pipeline
        '''

        config = "./PartSlip/GLIP/configs/glip_Swin_L.yaml"
        weight_path = "./PartSlip/models/glip_large_model.pth"
        
        print("[Module 2] 3/11. Loading GLIP model...")
        glip_demo = load_model(config, weight_path)

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        io = IO()
        os.makedirs(save_dir, exist_ok=True)
        
        print("[Module 2] 4/11 Normalizing input point cloud...")
        xyz, rgb = normalize_pc(input_pc_file, save_dir, io, device)
        
        print("[Module 2] 5/11 Rendering input point cloud...")
        img_dir, pc_idx, screen_coords = render_pc(xyz, rgb, save_dir, device)
        
        print("[Module 2] 6/11 Glip infrence...")
        preds = glip_inference(glip_demo, save_dir, part_names)
        
        print('[Module 2] 7/11 Generating superpoints...')
        superpoint = gen_superpoint(xyz, rgb, visualize=True, save_dir=save_dir)
        
        print('[Module 2] 8/11 Generating 3D part segmentations...')
        sem_seg, ins_seg = bbox2seg(xyz, superpoint, preds, screen_coords, pc_idx, part_names, save_dir, solve_instance_seg=True)
    
    def grounding_segmentation(self):
        '''
        Ground the segmentation results to the world coordinate system

        Returns: No return value, but create a self.grounded_seg_pcd
        '''
        print("[Module 2] 9/11 Grounding the segmentation results...")
        # Load the original point cloud as the base for the grounding
        original_pcd = o3d.io.read_point_cloud(self.point_cloud_path) # note that this is not a o3d.t
        original_points = np.asarray(original_pcd.points)
        
        parts_path = glob.glob(os.path.join(self.seg_output_dir, "semantic_seg", "*.ply"))

        # Initialize an empty list to hold the part_rank and color for each point
        all_part_ranks = np.zeros(shape=(original_points.shape[0], 1), dtype=np.int32)
        all_colors = np.zeros_like(original_points)
        
        for part_path in parts_path:
            # Load the point cloud
            pcd = o3d.io.read_point_cloud(part_path)
            
            # get the current color coding for this specific part. Only parts are colored with white
            colors = np.asarray(pcd.colors)

            # Check for colored points ([1, 1, 1] as color)
            colored_indices = np.all(colors != [0.0, 0.0, 0.0], axis=1)

            # Assign part rank and color to the corresponding points
            for is_colored in colored_indices:
                if is_colored:
                    all_part_ranks[colored_indices] = self.rank_dict[Path(part_path).stem]
                    all_colors[colored_indices] = self.colors_code[self.rank_dict[Path(part_path).stem]]

        # Create a new point cloud
        self.grounded_seg_pcd = o3d.t.geometry.PointCloud(original_points)

        self.grounded_seg_pcd.point.colors = o3c.Tensor(all_colors, dtype=o3c.Dtype.Float32)

        self.grounded_seg_pcd.point.part_rank = o3c.Tensor(all_part_ranks, dtype=o3c.Dtype.Int32)
        
        # useful when fusing the high gaussian gradient and segmentation results
        self.semantic_points = original_points
        self.semantic_part_rank = all_part_ranks

        o3d.t.io.write_point_cloud(os.path.join(self.pointcloud_folder, "grounded_segmentation.ply"), self.grounded_seg_pcd)

    def fuse_gaussian_and_segmentation(self):

        print("[Module 2] 10/11 Fusing the segmentation results with the high gaussian gradient points...")

        gaussian_pcd = o3d.t.io.read_point_cloud(os.path.join(self.output_folder, "high_grad_pts.pcd"))
        gaussian_points = gaussian_pcd.point.positions.numpy()


        gaussian_part_rank = np.zeros((gaussian_pcd.point.ranks.shape[0],1), dtype=np.int32)

        gaussian_color = np.zeros_like(gaussian_points)

        for i, target_point in enumerate(gaussian_points):
            # Compute the Euclidean distance from each point to the target_point
            distances = np.linalg.norm(self.semantic_points - target_point, axis=1)

            # Find the index of the closest point
            closest_point_index = np.argmin(distances)

            # Assign the semantic part rank of the closest point to the target point
            gaussian_part_rank[i] = self.semantic_part_rank[closest_point_index]
            gaussian_color[i] = self.colors_code[gaussian_part_rank[i]]

        gaussian_pcd.point.part_rank = o3c.Tensor(gaussian_part_rank, dtype=o3c.Dtype.Int32)
        gaussian_pcd.point.colors = o3c.Tensor(gaussian_color, dtype=o3c.Dtype.Float32)
        
        o3d.t.io.write_point_cloud(os.path.join(self.output_folder, "fused_gaussian_segmentation.ply"), gaussian_pcd)
        self.fused_pcd = gaussian_pcd

    def propose_next_best_touch(self):
        '''
        Propose the next best touch based on the fused point cloud
        '''
        print("[Module 2] Proposing the next best touch...")
        positions = self.fused_pcd.point.positions.numpy()
        part_rank = self.fused_pcd.point.part_rank.numpy().squeeze()
        rank = self.fused_pcd.point.ranks.numpy().squeeze()

        # Zip the lists and sort by part_rank first, then by gradient_rank
        # The awkward lambda function is used to handle the case where part_rank or gradient_rank is 0 (unassigned)
        sorted_coor = sorted(zip(positions, part_rank, rank), key=lambda pair: (pair[1] if pair[1] != 0 else float('inf'), pair[2] if pair[2] != 0 else float('inf')))


        # Step 2: Group by b values
        grouped_by_part_rank = {}
        for x, b_val, c_val in sorted_coor:
            if b_val not in grouped_by_part_rank:
                grouped_by_part_rank[b_val] = []
            grouped_by_part_rank[b_val].append(x)

        # Step 3: Select top 3 elements with highest-ranked b value (not 0)
        part_rank_values_sorted = sorted([key for key in grouped_by_part_rank if key != 0])

        selected_coor = []
        prioritized_rank = range(int(len(self.rank_dict.keys())*0.6))
        proposal_quota = 5 * len(prioritized_rank) + 5

        for rank in prioritized_rank:
            try:
                selected_coor.append(grouped_by_part_rank[part_rank_values_sorted[rank]][1])
                selected_coor.append(grouped_by_part_rank[part_rank_values_sorted[rank]][2])
                selected_coor.append(grouped_by_part_rank[part_rank_values_sorted[rank]][3])
                selected_coor.append(grouped_by_part_rank[part_rank_values_sorted[rank]][4])
                selected_coor.append(grouped_by_part_rank[part_rank_values_sorted[rank]][5])
            except IndexError:
                continue


        # Step 5: Select 3 more from overall ranked list, excluding previously selected
        remaining_list = [x for x, _, _ in sorted_coor if self._is_in_list(x, selected_coor)]

        need_to_fill = proposal_quota - len(selected_coor)

        additional_selection = []
        for i in range(need_to_fill):
            try:
                additional_selection.append(remaining_list[i])
            except IndexError:
                break

        # Combine the selections
        final_selection = selected_coor + additional_selection
        print("The next best touch points are:")
        for points in final_selection:
            print(points)
    
    def _partname_extraction_call(self, image_path, model_name="gpt-4o"):
        '''
        Extract the part names from the image using the VLM model, and rank them in terms of which one to touch first
        '''

        # Getting the base64 string
        base64_image = self._encode_image(image_path)

        PROMPT_MESSAGES = [
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "Take a deep breath. You are a very descriptive and helpful AI assistant. You will be given a picture with an object in the center. First, tell the classification of the object. Be as descriptive as possible. Then, You will need to describe the major parts that make up the object. Use label-like everyday single words. When giving the label, think of parts that are difficult to have a good perception of purely based on vision but would also rely on tactile perception. For example, a small button on an earphone case. Also, rank the parts in terms of which one to touch first."
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
        
        response = self._call_openai_api(PROMPT_MESSAGES, model_name)

        if response is None:
            print("Failed to get a response from OpenAI API")
            return None, None
        
        return response.classification, response.parts
    
    def _encode_image(self, image_path):
        '''
        Encode the image to base64 string for OpenAI API
        '''
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _call_openai_api(self, prompt_messages, model_name) -> Union[PartResponse, None]:
        '''
        Wrapper function to call OpenAI API
        '''
        params = {
            "model": model_name,
            "messages": prompt_messages,
            "max_tokens": 400,
            "temperature": 0,
            "response_format": PartResponse,
        }
        result = self.OAIclient.beta.chat.completions.parse(**params)
        return result.choices[0].message.parsed

    def _create_rank_dict(self, parts):
        '''
        Create a dictionary to rank the parts in terms of which one to touch first.
        Also create a list for color coding the parts
        '''
        self.rank_dict = {}
        for i, part in enumerate(parts):
            self.rank_dict[part] = i+1 # rank starts from 1. 0 is reserved for unclassified points
        self.colors_code = np.random.rand(len(parts)+1, 3)
        self.colors_code[0] = [0, 0, 0] # the first color is black, which is for all the unclassified points
    
    def _is_in_list(self, arr, arr_list):
        return any(np.array_equal(arr, x) for x in arr_list)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default=None, required=True, choices=["touch", "partname"], help="Operation to perform. Options: touch, partname")
    parser.add_argument("--data_name", type=str, default="transparent_bunny", help="Name of the dataset folder")
    parser.add_argument("--model_name", type=str, default="9view", help="Name of the model. It will impact the output and eval folder name. You can technically name this whatever you want.")
    parser.add_argument("--mesh_name", type=str, default="bunny", help="Name of the mesh file")
    parser.add_argument("--object_name", type=str, default=None, help="Name of the object")
    parser.add_argument("--part_name", type=str, nargs='+', default=None, help="List of part names")
    parser.add_argument("--llm_name", type=str, default='gpt-4o', help="Name of the LLM model. Currently, only OpenAI API is supported.")
    parser.add_argument("--verbose", type=bool, default=False, help="False: Only show important logs. True: Show all logs.")
    
    args = parser.parse_args()
    mode = args.mode
    data_name = args.data_name
    verbose = args.verbose
    img_folder_path = Path(f"datasets/{data_name}/images/")
    vlm = VLM(img_folder_path)

    if mode == "touch":
        model_name = args.model_name
        object_name = args.object_name
        part_name = args.part_name
        mesh_name = args.mesh_name

        output_folder_path = Path(f"outputs/{data_name}/{model_name}/")
        mesh_path = Path(f"outputs/{data_name}/{model_name}/MESH/{mesh_name}.ply")

        vlm.update_output_folder(output_folder_path)
        vlm.touch_selection(object_name=object_name, part_name=part_name, mesh_path=mesh_path)
            
    elif mode == "partname":
        model_name = args.model_name
        vlm.partname_extraction()

if __name__ == "__main__":
    main()
