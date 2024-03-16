import numpy as np
import torch

def features_to_PC(features, K, b):
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

    return np.stack([x,y,z], axis=-1)

def get_seeing_mask(features, d_min = 1, d_max = 40):
    present_mask = np.all(features != -1, axis=1)

    d = features[present_mask, 0] - features[present_mask, 2]
    d_mask = (d_min < d) & (d < d_max)

    present_mask[present_mask.copy()] = d_mask
    return present_mask

def features_to_PC_torch(features, K, b):
    fsu = K[0, 0]
    fsv = K[1, 1]
    cu = K[0, 2]
    cv = K[1, 2]

    Ul = features[..., 0]
    Vl = features[..., 1]
    Ur = features[..., 2]
    Vr = features[..., 3]

    V = (Vl + Vr) / 2
    d = Ul - Ur
    z = fsu * b / d
    x = (Ul - cu) * z / fsu
    y = (V - cv) * z / fsv

    return torch.stack([x, y, z], dim=-1)

def get_seeing_mask_torch(features, d_min=1, d_max=40):
    present_mask = torch.all(features != -1, dim=1)

    d = features[present_mask, 0] - features[present_mask, 2]
    d_mask = (d > d_min) & (d < d_max)

    present_mask[present_mask.clone()] = d_mask
    return present_mask
