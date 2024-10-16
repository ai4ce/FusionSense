**More Documentation Ongoing for VLM Reasoning and Real World Experiments. The README Needs a Lot of Cleaning and Update**

:new: [2024-10-17] *Installation for Hardware Integration/3D Printing Updated.*

:new: [2024-10-15] *Installation for Robotics Software Updated.*

:new: [2024-10-11] *Made Public*
# FusionSense
### [[Page](https://ai4ce.github.io/FusionSense/)] | [[Paper](https://arxiv.org/abs/2410.08282)] | [[Video](https://youtu.be/thC0PeAQxe0)]
This is the official implementation of [FusionSense: Bridging Common Sense, Vision, and Touch for Robust Sparse-View Reconstruction](https://ai4ce.github.io/FusionSense/)


[Irving Fang](https://irvingf7.github.io/), [Kairui Shi](https://kairui-shi.github.io/), [Xujin He](https://www.linkedin.com/in/kim-he-064a36258/), Siqi Tan, Yifan Wang, [Hanwen Zhao](https://www.linkedin.com/in/hanwen-zhao-2523a4104/), [Hung-Jui Huang](https://joehjhuang.github.io), [Wenzhen Yuan](https://scholar.google.com/citations?user=SNqm6doAAAAJ&hl=en), [Chen Feng](https://scholar.google.com/citations?user=YeG8ZM0AAAAJ), [Jing Zhang](https://jingz6676.github.io)

FusionSense is a novel 3D reconstruction framework that enables robots to fuse priors from foundation models with highly sparse observations from vision and tactile sensors. It enables visually and geometrically accurate scene and object reconstruction, even for conventionally challenging objects.

<img src="assets/snapshot.gif" alt="FusionSense Snapshot" width="200"/>

## Preparation 

### Step 0: Install Everything Robotics
We used a depth camera mounted on a robot arm powered by `ROS2` to acquire pictures with accurate pose information. We also used a tactile sensor for <b>Active Touch Selection</b>.

If you have no need for this part, feel free to jump into [Step 1](https://github.com/ai4ce/FusionSense/blob/main/README.md#step-1-install-3d-gaussian-dependencies-and-nerfstudio) for the 3D Gaussian pipeline of <b>Robust Global Shape Representation</b> and <b>Local Geometric Optimization</b>.

- For installing robotics software, please see [Robotics Software Installation](./instructions/install_robotics.md). 
- For hardware integration, please see [3D Printing Instructions](./instructions/3d_printing.md).

### Step 1: Install 3D Gaussian Dependencies and Nerfstudio
**Note:** Because our major dependencies, `Nerfstudio` and `Grounded-SAM-2`, officially support two different CUDA versions (11.8 vs. 12.1), we will have to create two separate environments. We hope to resolve this in the future when `Nerfstudio` bumps its official CUDA support version.

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

Build the environment
```sh
pip install -e .
```

### Step 3: Install Grounded-SAM-2

We use `Grounded-SAM-2` for segmenting the foreground and background. Please make sure to use our modified submodule. 

We recommend starting a separate Conda environment since `Grounded-SAM-2` requires CUDA 12.1, which is not yet officially supported by `Nerfstudio`.
```sh
cd Grounded-SAM2-for-masking
cd checkpoints
bash download_ckpts.sh
cd ../gdino_checkpoints
bash download_ckpts.sh
```



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
Set your scene path and prompt text with an '.' at the end.   
`eg. 'transparent white statue.'`   

```bash
conda activate G-SAM-2
cd Grounded-SAM2-for-masking
python grounded_sam2_hf_model_imgs_MaskExtract.py  --path {ABSOLUTE_PATH} --text {TEXT_PROMPT_FOR_TARGET_OBJ}
cd ..
```

run the script to extract masks.   

If the `num_no_detection` is not 0, you need to select the frame again. Then you will see mask_imgs in `/masks`, and you can check `/annotated` frames to see the results more directly.  

### Run pipeline
You can change configs here: `configs/config.py`
```sh
conda activate fusionsense
python scripts/train.py --data_name {DATASET_NAME} --model_name {MODEL_NAME} --configs {CONFIG_PATH}
```

### Render outputs

For render jpeg or mp4 outputs using nerfstudio, we recommend install ffmpeg in conda environment:

```sh
conda install -c conda-forge x264=='1!161.3030' ffmpeg=4.3.2
```

To render outputs of pretrained models:

```sh
python scripts/render_video.py camera-path --load_config your-model-config --camera_path_filename camera_path.json --rendered_output_names rgb depth normal
```
more details in nerfstudio `ns-render`.

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
    │   ├── rgb_1.png
    │   └── rgb_2.png
    │ 
    ├── realsense_depth/
    │   ├── depth_1.png
    │   └── depth_2.png
    │
    │── tactile/
    │   ├── image
    │   ├── mask
    │   ├── normal
    │   └── patch
    │
    ├── model.stl       # need for evaluation
    │
    ├── normals_from_pretrain/ # generated
    │   ├── rgb_1.png
    │   └── rgb_2.png
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
    │   └── mesh.ply
    │
    ├── nerfstudio_models/
    │   └── 30000.ckpt
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
