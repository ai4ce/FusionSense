**Note 1:** Because our major dependencies, `Nerfstudio` and `Grounded-SAM-2`, officially support two different CUDA versions (11.8 vs. 12.1), we will have to create two separate environments. We hope to resolve this in the future when `Nerfstudio` bumps its official CUDA support version.

**Note 2:** We use `Grounded-SAM-2` for segmenting the foreground and background. Please make sure to use our modified submodule. 

We recommend starting a separate Conda environment since `Grounded-SAM-2` requires CUDA 12.1, which is not yet officially supported by `Nerfstudio`.

Starting from the root of this repo, first download the checkpoints needed.
```sh
cd Grounded-SAM2-for-masking
```
```sh
cd checkpoints
```
```sh
bash download_ckpts.sh
```
```sh
cd ../gdino_checkpoints
```
```sh
bash download_ckpts.sh
```

Then we create an environment for this part.
```sh
conda create -n G-SAM-2 python=3.10
```
```sh
conda activate G-SAM-2
```

We then install `PyTorch 2.3.1` and its friends
```sh
conda install pytorch==2.3.1 torchvision==0.18.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
```sh
pip install opencv-python supervision transformers
```
and then `CUDA 12.1` development kit as we will need it to compile `Deformable Attention` operator used in `Grounded-SAM-2`.
```sh
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
```
Use `which nvcc`to check that the installation is successful. The result should look like
```
/home/irving/miniconda3/envs/G-SAM-2/bin/nvcc
```
Then, the `CUDA_HOME` should be set to the following. (Modify according to your output from the last step)
```sh
export CUDA_HOME=/home/irving/miniconda3/envs/G-SAM-2/
```
Install Segment Anything 2
```sh
pip install -e . 
```
Install Grounding DINO. Yes this is not a typo. Grounding DINO is needed to run `Grounded-SAM-2`.
```sh
pip install --no-build-isolation -e grounding_dino
```

If you encounter any problem, you can check out `Grounded-SAM2-for-masking`'s official [installation guide](https://github.com/IDEA-Research/Grounded-SAM-2#installation).