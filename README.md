# Instructions

## Environment Setup
- Python 3.7 to 3.10 should work. In our application, we use Python=3.10
- create the designated environment recomd: conda create -n recomd Python=3.10
- Activate the environment: conda activate recomd
- GPU CUDA version: CUDA is usually installed system side not conda environment wide. My CUDA version = 12.0

```
conda install pytorch torchvision torchaudio -c pytorch
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv
pip install pandas ipykernel matplotlib seaborn
``` 
- 