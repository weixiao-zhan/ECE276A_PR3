import numpy as np
from pr3_utils import *
from stereo import *

datasets = ["03", "10"]
for dataset in datasets:
    time_stamp,features,linear_velocity,angular_velocity,k,b,imu_T_cam = \
        load_data(f"../data/{dataset}.npz")

    num_time_stamp = time_stamp.shape[0]
    num_features = features.shape[1]

    tau = time_stamp[1:] - time_stamp[:-1]
    velocity = np.concatenate([linear_velocity, angular_velocity], axis=1)

    # noise model
    W = 1e-3

    # init
    T_mean = np.zeros([time_stamp.shape[0], 4, 4])
    T_covar = np.zeros([time_stamp.shape[0], 6, 6])
    T_mean[0,:,:] = np.array([
        [1, 0, 0,0],
        [0, 1, 0,0],
        [0, 0, 1,0],
        [0, 0, 0,1],
    ])
    T_covar[0,:,:] = np.diag([0,0,0,0,0,0])

    # EKF predict
    for t in range(1, num_time_stamp):
        T_mean[t,:,:] = T_mean[t-1,:,:] @ twist2pose(tau[t-1]*axangle2twist(velocity[t]))
        F = twist2pose(-tau[t-1]*axangle2adtwist(velocity[t]))
        T_covar[t,:,:] = F @ T_covar[t-1,:,:] @ F.T + W*np.eye(6)

    fig,_ = visualize_trajectory("EKF_localization", T_mean, 100*T_covar)
    fig.savefig(f'../img/{dataset}_EKF_localization.png', dpi=300)
    plt.show()

    np.save(f"../data/{dataset}_EKF_localization_T_mean",T_mean)
    np.save(f"../data/{dataset}_EKF_localization_T_covar",T_covar)


