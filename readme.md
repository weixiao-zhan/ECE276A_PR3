# File Description

### helper files
[`code/pr3_utils.py`](code/pr3_utils.py): load data, Lie algebra, plot

[`code/pr3_torch_utils.py`](code/pr3_torch_utils.py): reimplement `code/pr3_utils.py` using PyTorch instead on NumPy, so that I can use GPU to accelerate SLAM.

[`code/stereo.py`](code/stereo.py): code for computing 3d coordinate from stereo camera pixel coordinate

### EKF code
[`code/EKF_localization.py`](code/EKF_localization.py):
read in data, build EKF localization trajectory, save trajectory in `data/{dataset}_EKF_localization_T_mean.npy`. 
(using NumPy)

[`code/EKF_mapping_init.py`](code/EKF_mapping_init.py):
read in data and localization T_mean, build EKF mapping of feature points in 3d. 
Depends on the preprocessing selection, save init coordinates in `data/{dataset}_EKF_mapping_M_init_all.npy`,
`data/{dataset}_EKF_mapping_M_init_selected.npy`,
and `data/{dataset}_EKF_mapping_M_mask_selected.npy`. 
(using NumPy)

[`code/EKF_SLAM.py`](code/EKF_SLAM.py):
read in data, perform SLAM over sampled features. (using PyTorch)

[`code/EKF_SLAM_TW.py`](code/EKF_SLAM_TW.py):
read in data, perform time window SLAM over sampled features. (using PyTorch)

**Note**
1. Note1: `*.py` are automatically converted from `*.ipynb` files.
 Feel free to run those jupyter files instead.

2. PyTorch code should work on CPU, MPS, CUDA devices. 
However MPS only support `float32`, which may cause over floating `nan`.
Use `CPU + float64` or `CUDA + float64` instead.

### [`img/`](img/)
full size img that are present in report.