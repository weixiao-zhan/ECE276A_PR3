import numpy as np

def features_to_PC(features, K, b, opt=False):
    fsu = K[0,0]
    fsv = K[1,1]
    cu  = K[0,2]
    cv  = K[1,2]

    Ul = features[..., 0]
    Vl = features[..., 1]
    Ur = features[..., 2]
    Vr = features[..., 3]

    V = (Vl+Vr)/2
    d = Ul - Ur
    z = fsu * b / d
    x = (Ul - cu)*z / fsu
    y = (V  - cv)*z / fsv
    if opt:
        PC_camera = np.stack([x,y,z], axis=-1)
    else:
        PC_camera = np.stack([z, -x, -y], axis=-1)
    return PC_camera

def get_seeing_mask(features, d_min = 1, d_max = 40):
    present_mask = np.all(features != -1, axis=1)

    d = features[present_mask, 0] - features[present_mask, 2]
    d_mask = (d_min < d) & (d < d_max)

    present_mask[present_mask] = d_mask
    return present_mask