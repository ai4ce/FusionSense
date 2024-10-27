**Note:** Because our major dependencies, `Nerfstudio` and `Grounded-SAM-2`, officially support two different CUDA versions (11.8 vs. 12.1), we will have to create two separate environments. We hope to resolve this in the future when `Nerfstudio` bumps its official CUDA support version.


Clone our repo. Make sure to clone the submodule as well by using `--recursive`.
```sh
git clone --recursive https://github.com/ai4ce/FusionSense.git
```

Create the environment.
```sh
cd FusionSense
```
```sh
conda env create -f config.yml
```
```sh
conda activate fusionsense
```

Install compatible **PyTorch** and **cuda-toolkit** version:

```sh
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```
```sh
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

Install **mmcv**:

```sh
pip install mmcv
```

Install **tinycudann**:

```sh
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Build the environment
```sh
pip install -e .
```
Note that this part of our codebase is largely modified from `dn-splatter`, so we did not modify their project name out of respect.

This environment is largely a mix of `dn-splatter`([doc](https://github.com/maturk/dn-splatter?tab=readme-ov-file#installation)) and `Metric3D`([doc](https://github.com/YvanYin/Metric3D/tree/main)). If you encounter any installation problem, in addition to posting an issue in this repo, you are welcome to checkout their repos as well.
