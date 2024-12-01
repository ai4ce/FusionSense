**More Documentation Ongoing for VLM Reasoning and Real World Experiments. This README is Still Being Actively Updated**

:new: [2024-11-15] *Installation for VLM Reasoning & Active Touch Selection Updated.*

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
This repo has been tested on Ubuntu `20.04` and `22.04`. The real-world experiment is conducted on `22.04` as `ROS2 Humble` requires it.

### Step 0: Install Everything Robotics
We used a depth camera mounted on a robot arm powered by `ROS2` to acquire pictures with accurate pose information. We also used a tactile sensor for <b>Active Touch Selection</b>.

If you have no need for this part, feel free to jump into [Step 1](https://github.com/ai4ce/FusionSense/blob/main/README.md#step-1-install-3d-gaussian-dependencies-and-nerfstudio) for the 3D Gaussian pipeline of <b>Robust Global Shape Representation</b> and <b>Local Geometric Optimization</b>.

- For installing robotics software, please see [Robotics Software Installation](./instructions/install_robotics.md). 
- For hardware integration, please see [3D Printing Instructions](./instructions/3d_printing.md).

**Note:** `ROS2` doesn't play well with Conda in general. See [official doc](https://docs.ros.org/en/jazzy/How-To-Guides/Using-Python-Packages.html) and [this issue in the ROS2 repo](https://github.com/ros2/ros2/issues/1094). As a result, in this project, `ROS2` uses the minimal system Python environment and have limited direct interaction with the Python perception modules.

### Step 1: Install 3D Gaussian Dependencies
We will need two independent virtual environments due to some compatibility issue. 
#### Step 1.1: DN-Splatter and Metric3D
Please see [DN-Splatter and Metric3D Installation](./instructions/dn_splatter_metric_3d.md)

#### Step 1.2: Grounded-SAM-2
Please see [Grounded-SAM-2 Installation](./instructions/grounded_sam_2.md)

### Step 2: Install VLM Dependencies for Active Touch Selection

Please see [Active Touch Selection Installation](instructions/active_touch.md)

## Usage
### 1. Robust Global Shape Representation
#### a. Prepare Data
You can see [here](https://huggingface.co/datasets/ai4ce/FusionSense) for an example dataset structure.

Note that a lot of the folders are generated during the pipeline. The data needed to start this projects are: `images`, `realsense_depth`, and `transforms.json`.

The ROS2 packages I shared can be used to acquire the aforementioned data. Or you can manually format your own dataset this way.

The project assume that all the folders in the HuggingFace repo are put under `FusionSense/datasets/`.
#### b. Extract Mask

<details>
<summary>
If you want to let VLM classify the object, click this line. 

If you want to manually specify the name, please read ahead.</summary>

Inside our main conda env
```bash
conda activate fusionsense
```
Run this script.
```bash
python scripts/VLM.py --mode partname --data_name {DATASET_NAME}
```
- `mode`: Operation mode of our VLM pipeline. In this case we want it to give us partname.
- `data_name`: Name of the specific dataset folder. Example: transparent_bunny
</details>


Whether you got the name from VLM or your own brain, we can proceed with that.

Switch your conda env first
```bash
conda activate G-SAM-2
```
Inside the submodule of our Grounded-SAM2
```bash
cd Grounded-SAM2-for-masking
```
Run the script to extract masks by setting your dataset path and object name prompt text. The prompt text ends with an '.' at the end. 

You can use something you came up with, or one proposed by the VLM. In our experience, both works fine.

`eg. --path /home/irving/FusionSense/dataset/transparent_bunny --text 'transparent bunny statue.'`   
```bash
python grounded_sam2_hf_model_imgs_MaskExtract.py  --path {ABSOLUTE_PATH} --text {TEXT_PROMPT_FOR_TARGET_OBJ}
```

You will see mask_imgs in the newly created `/masks` folder, and you can check `/annotated` folder to see the results more directly.

#### c. Select Frames
set `train.txt` with images id.
You can pick images that have better masking for better final result. Although in our experiment we didn't cherrypick which images to use except that we want images to be relatively evenly spread out.

#### d. Run Pipeline
This pipeline is mostly run in `Nerfstudio`.
You can change configs at `configs/config.py`
First go back to our main conda environment and main folder
```sh
conda activate fusionsense
```
```sh
cd ..
```
Then we run
```sh
python scripts/train.py --data_name {DATASET_NAME} --model_name {MODEL_NAME} --load_touches {True, False} --configs {CONFIG_PATH} --verbose {True, False} --vram_size {"large", "small"}
```
- `data_name`: Name of the dataset folder
- `model_name`: Name of the model you train. It will impact the output and eval folder name. You can technically name this whatever you want.`
- `load_touches`: Whether to load tactile data. Default=False
- `configs`: Path to the Nerfstudio config file
- `verbose`: False: Only show important logs. True: Show all logs. Default=False
- `vram_size`: "large" or "small". Decides the foundation models variants used in the pipeline. Default="large"

An example using the provided data would be:
```sh
python scripts/train.py --data_name transparent_bunny --model_name 9view --configs configs/config.py --vram_size small
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

