import numpy as np

def features_to_PC_camera(features, K, b, d_min = 1, d_max = 40):
    fsu = K[0,0]
    fsv = K[1,1]
    cu  = K[0,2]
    cv  = K[1,2]

    present_mask = np.all(features != -1, axis=1)
    Ul = features[present_mask, 0]
    Vl = features[present_mask, 1]
    Ur = features[present_mask, 2]
    Vr = features[present_mask, 3]

    V = (Vl+Vr)/2
    d = Ul - Ur

    d_mask = (d_min < d) & (d < d_max)
    mask_combined = np.zeros(features.shape[0], dtype=bool)
    mask_combined[present_mask] = d_mask

    z = fsu * b / d[d_mask]
    x = (Ul[d_mask]*z - cu) / fsu
    y = (V[d_mask]*z - cv)/ fsv

    re_PC_camera = np.zeros([features.shape[0], 3])
    re_PC_camera[mask_combined] = np.stack([z,-x,-y], axis=-1)

    return re_PC_camera, mask_combined

def get_M_init(features, T, K, b, imu_T_cam):
    seen_mask = np.zeros(features.shape[1], dtype=bool)
    features_init = np.zeros([features.shape[1], 3])

    for t in range(T.shape[0]):
        PC_camera, present_mask = features_to_PC_camera(features[t], K, b)
        PC_camera_hom = np.hstack([PC_camera, np.ones([PC_camera.shape[0],1])])
        PC_world_hom  = (T[t] @ imu_T_cam @ PC_camera_hom.T).T
        PC_world = PC_world_hom[:,:3]

        first_time_mask = np.logical_and((np.logical_not(seen_mask)), present_mask)
        features_init[first_time_mask] = PC_world[first_time_mask]

        seen_mask[first_time_mask] = True
        if np.all(seen_mask):
            print(f"seen all features at {t} stamp")
            break
    
    return features_init, seen_mask