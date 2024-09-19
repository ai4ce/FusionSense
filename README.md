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
### Select Frames
set `train.txt` with images id.

### Extract Mask
**Switch your conda env first**  
set your scene path and prompt text with the end of '.'   
`eg. 'transparent white statue.'`   

```bash   
conda activate G-SAM-2
cd Grounded-SAM2-for-masking
python grounded_sam2_hf_model_imgs_MaskExtract.py  --path {ABSOLUTE_PATH} --text {TEXT_PROMPT_FOR_TARGET_OBJ}
cd ..
```   
run the script to extract masks.   

If the `num_no_detection` is not 0, you need to select the frame again. Then you will see mask_imgs in `/masks`, and you can check `/annotated` frames to see the results more directly.  

### Run pipeline
```sh
conda activate fusionsense
python Initial_Recon.py --data_name {DATASET_NAME} --model_name {MODEL_NAME}
```

## Dataset Format
```bash
datasets/
    ds_name/
    │
    ├── transforms.json # need for training
    │
    ├── train.txt
    │
    ├── images/
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
    ├── model.stl       # need for evaluation
    │
    ├── normals_from_pretrain/ # generated
    │   ├── rgb_1.png
    │   └── rgb_2.png
    │
    ├── foreground_pcd.ply
    │
    └── merged_pcd.ply
```

## Outputs Format
```bash
outputs/
    ds_name/
    │
    ├── MESH/
    │   └── mesh.ply
    │
    ├── nerfstudio_models/
    │   └── 30000.ckpt
    │   
    ├── cluster_centers.npy
    │
    ├── config.yml
    │
    ├── high_grad_pts.pcd
    │
    ├── high_grad_pts_ascii.pcd
    │
    └── dataparser_transforms.json

eval/
    ds_name/ *evaluation results files*
```

## Render your video 
```bash
python render_video.py camera-path --load_config your-model-config --camera_path_filename camera_path.json
```
more details in nerfstudio `ns-render`.