# Active Touch Selection Installation

We are using [PartSlip](https://colin97.github.io/PartSLIP_page/) as our part segmentation model. As we have mentioned in the paper, the model itself can be substituted with more advanced work such as [Point-Sam](https://point-sam.github.io/). 

**Important**: If you stick with PartSlip, please use our modified version included in the repo instead of the official implementation. The official implementation is incompatible with more modern CUDA and PyTorch due to the deprecation of `TH/TH(C)` namespace and the introduction of `ATen` into PyTorch.

We will install everything in the `fusionsense` environment.

## Prerequisites

```sh
conda activate fusionsense
```
```sh
conda install boost eigen
```
```sh
pip install yacs nltk inflect einops prettytable ftfy openai
```

## PyTorch3D
Install PyTorch3D. 
This specific installation method seems most reliable.
```sh
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

## Checkpoint Download
Then, we get into the PartSlip folder to manually compile a few things
```sh
cd PartSlip
```
Download the pre-trained model we need.
```sh
bash download_ckpts.sh
```

## GLIP
Then we install GLIP, a key dependency.
```sh
cd GLIP
```
```sh
python setup.py build develop --user
```

## `cut-pursuit`
Finally, we compile the [cut-pursuit](https://github.com/loicland/superpoint_graph) for computing superpoints.
```sh
CONDAENV=YOUR_CONDA_ENVIRONMENT_LOCATION
```
An example is
```sh
CONDAENV=/home/irving/miniconda3/envs/fusionsense
```

We are ready to compile the package.
```sh
cd ../partition/cut-pursuit
```
```sh
mkdir build && cd build
```
```sh
cmake .. -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.8.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.8 -DBOOST_INCLUDEDIR=$CONDAENV/include -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3
```
```sh
make
```

After this, we should have everything ready for perform Active Touch Selection.
