# FusionSense
Integrates the vision, touch, and common-sense information of foundational models, customized to the agent's perceptual needs.

## Installation

### Step 1: Install dependencies and Nerfstudio

```sh
git clone --recursive https://github.com/ai4ce/FusionSense.git
cd FusionSense
conda env create -f config.yml
conda activate fusionsense
```

Install compatible **pytorch** and **cuda-toolkit** version:

```sh
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

Install **tinycudann**:

```sh
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Step 2: Build Fusionsense

```sh
pip install -e .
```

### Step 3: Install and Run Grounded SAM (Need to switch virtual env)

We use Grounded-SAM2 for segmenting the foreground and background. For each 

```sh
cd Grounded-SAM2-for-masking
cd checkpoints
bash download_ckpts.sh
cd ../gdino_checkpoints
bash download_ckpts.sh
```

We recommend starting a separate Conda environment for Grounded-SAM2, since Grounded-SAM2 requires cuda=12.1

```sh
conda create -n G-SAM-2
conda activate G-SAM-2
conda install pip 
conda install opencv supervision transformers
pip install torch torchvision torchaudio
# select cuda version 12.1
export CUDA_HOME=/path/to/cuda-12.1/
# install Segment Anything 2
pip install -e . 
# install Grounding DINO
pip install --no-build-isolation -e grounding_dino
```

For further installation problems:

- For `dn-splatter`, see [Installation](https://github.com/maturk/dn-splatter?tab=readme-ov-file#installation)   

- For `Grounded-SAM2-for-masking`, see [Installation](https://github.com/IDEA-Research/Grounded-SAM-2#installation)

## Usage

1. **Select frames**:  

    Run `delete.py` to select frames you want, or manually select, and you will get a folder of selected frames and transforms.json.  

    ```bash
    python select_imgs.py --path datasets/{PATH}
    ```
    Select frames you want, set your selected images in **train.txt**.

    **Remember to set `transforms.json` in right format.**

2. **Generate Mask_imgs by [Grounded_SAM_2](https://github.com/IDEA-Research/Grounded-SAM-2)**:   

    **Switch your conda env first**  

    set your scene path and prompt text with the end of '.'   
    `eg. 'transparent white statue.'`   

    ```bash   
    cd Grounded-SAM2-for-masking
    python grounded_sam2_hf_model_imgs_MaskExtract.py  --path {ABSOLUTE_PATH} --text {TEXT_PROMPT_FOR_TARGET_OBJ}
    ```   
    run the script to extract masks.   

    If the `num_no_detection` is not 0, you need to select the frame again. Then you will see mask_imgs in `path/masks`, and you can check `path/annotated` frames to see the results more directly.   
    
3. **Generate VisualHull by masks and transforms.json**:  

    run `VisualHull.py` to generate visual hull.  
    ```bash  
    python VisualHull.py --path your-path  
    ```
    
    You will get a point cloud file `foreground_pcd.ply`, and a screenshot `voxels.png` of checking whether the generated VisualHull is correct.    

    Also, you can run this scirpt setting the voxel grid resolution to 0.002 to get a more detailed point cloud as `object.ply` for hull pruning.

    <img src="assets/voxels.png" width="300">

4. **RealSense depth & [Metric3Dv2](https://github.com/YvanYin/Metric3D) depth**:  

    Get your realsense depth from your camera file in `realsense_depth` folder.  

    Use your RGB images to generate predict depth with Metric3Dv2.  
    ```bash
    python run_metric3d_depth.py --root_dir your-path
    ```
    <u>**Remember to set your camera intrinsics and image size in that file**</u>   

5. **Generate initial GS model sparse points**:  

    run the script to generate initial sparse points using VisualHull pcd as forground and Metric3Dv2 depth as background.    
    ```bash
    python generate_pcd.py --path your-path   
    ```

    The initial points will be saved in `path/merged_pcd.ply`  

6. **Generate normals by dsine**:

    set your rgb images path to generate normals.  
    ```bash
    python dn_splatter/scripts/normals_from_pretrain.py --data-dir [PATH_TO_DATA] --model-type dsine  
    ```

7. **Set transforms and configs**:

    To use realsense depth, set `"depth_file_path": "realsense_depth/depth_0.png"` each frame     

    To use initial pts, set `"ply_file_path": "merged_pcd.ply"`     

    To use Visual Hull prune supervised method, set `"object_pc_path": "foreground_pcd.ply"`    

8. **Train**:

    Select your method and configs.
    ```bash
    ns-train dn-splatter --pipeline.model.use-depth-loss True\
                        --pipeline.model.normal-lambda 0.4\
                        --pipeline.model.sensor-depth-lambda 0.2\
                        --pipeline.model.use-depth-smooth-loss True \
                        --pipeline.model.use-normal-loss True\
                        --pipeline.model.normal-supervision mono\
                        --pipeline.model.random_init False normal-nerfstudio\
                        --data your-path\
                        --load-pcd-normals True --load-3D-points True  --normal-format opencv
    ```

    To Train with touches:
    
    ```bash
    ns-train dn-splatter --pipeline.model.use-depth-loss True\
                        --pipeline.model.normal-lambda 0.4\
                        --pipeline.model.sensor-depth-lambda 0.2\
                        --pipeline.model.use-depth-smooth-loss True \
                        --pipeline.model.use-normal-loss True\
                        --pipeline.model.normal-supervision mono\
                        --pipeline.model.random_init False normal-nerfstudio\
                        --data your-path\
                        --load-touches True
                        --load-pcd-normals True --load-3D-points True  --normal-format opencv
    ```

    **If you want to load checkpoints**:
    ```bash
        ns-train dn-splatter 
                        --load-dir PATH_TO_CONFIG\
                        --pipeline.model.use-depth-loss True\
                        --pipeline.model.normal-lambda 0.4\
                        --pipeline.model.sensor-depth-lambda 0.2\
                        --pipeline.model.use-depth-smooth-loss True \
                        --pipeline.model.use-normal-loss True\
                        --pipeline.model.normal-supervision mono\
                        --pipeline.model.random_init False normal-nerfstudio\
                        --data your-path\
                        --load-pcd-normals True --load-3D-points True  --normal-format opencv
    ```

    To prepare touch-gs dataset:
    ```bash
        cd datasets/touchgs; python gs_to_ours_script.py; cd ../..
    ```
9. **Mesh Extraction**:
    ```python
    gs-mesh {dn, tsdf, sugar-coarse, gaussians, marching} --load-config [PATH] --output-dir [PATH]
    ```

10. **Export GSplat**:
    ```bash
    ns-export gaussian-splat --load-config outputs/unnamed/dn-splatter/2024-09-02_203650/config.yml --output-dir exports/splat/ 
    ```

## Dataset Format
```bash
tr-rabbit/
│
├── transforms.json
│
├── images/
│   ├── rgb_1.png
│   └── rgb_2.png
│
├── normals_from_pretrain/ # normal 
│   ├── rgb_1.png
│   └── rgb_2.png
│
├── realsense_depth/
│   ├── depth_1.png
│   └── depth_2.png
│
│── tactile/
│   ├── image
│   ├── mask
│   ├── normal
│   └── patch
│
├── object.ply
└── merged_pcd.ply
```