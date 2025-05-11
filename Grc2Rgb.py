import numpy as np
from matplotlib import pyplot as plt

c1_data_t = np.load('Data/c1_td_granules.npy')

c2_data_t = np.load('Data/c2_td_granules.npy')

c1_data_p = np.load('Data/c1_ps_granules.npy')

c2_data_p = np.load('Data/c2_ps_granules.npy')


def split_into_rgb(data):
    # data shape: (105, 118, 100, 2)
    total_channels = data.shape[1]
    group_size = total_channels // 3
    remainder = total_channels % 3

    # Calculate the actual size of each group
    sizes = [group_size + (1 if i < remainder else 0) for i in range(3)]

    # Obtain split index
    split_indices = np.cumsum(sizes)[:-1]

    group1, group2, group3 = np.split(data, split_indices, axis=1)

    return group1, group2, group3

c1_R_t, c1_G_t, c1_B_t = split_into_rgb(c1_data_t)
c2_R_t, c2_G_t, c2_B_t = split_into_rgb(c2_data_t)

c1_R_p, c1_G_p, c1_B_p = split_into_rgb(c1_data_p)
c2_R_p, c2_G_p, c2_B_p = split_into_rgb(c2_data_p)

c1_R_t = np.mean(c1_R_t, axis=1)
c1_G_t = np.mean(c1_G_t, axis=1)
c1_B_t = np.mean(c1_B_t, axis=1)
c2_R_t = np.mean(c2_R_t, axis=1)
c2_G_t = np.mean(c2_G_t, axis=1)
c2_B_t = np.mean(c2_B_t, axis=1)

c1_R_p = np.mean(c1_R_p, axis=1)
c1_G_p = np.mean(c1_G_p, axis=1)
c1_B_p = np.mean(c1_B_p, axis=1)
c2_R_p = np.mean(c2_R_p, axis=1)
c2_G_p = np.mean(c2_G_p, axis=1)
c2_B_p = np.mean(c2_B_p, axis=1)


def Gr2Rgb(time_data, phase_data, img_size=224):
    trials, windows, _ = time_data.shape
    output = np.zeros((trials, img_size, img_size), dtype=np.uint8)

    # extract min/max
    time_min_vals = time_data[..., 0]
    time_max_vals = time_data[..., 1]
    phase_min_vals = phase_data[..., 0]
    phase_max_vals = phase_data[..., 1]

    # normalization
    time_all = np.concatenate([time_min_vals.flatten(), time_max_vals.flatten()])
    phase_all = np.concatenate([phase_min_vals.flatten(), phase_max_vals.flatten()])

    time_min_global = time_all.min()
    time_max_global = time_all.max()
    phase_min_global = phase_all.min()
    phase_max_global = phase_all.max()

    # normalization to [0, 1]
    time_min_norm = (time_min_vals - time_min_global) / (time_max_global - time_min_global + 1e-8)
    time_max_norm = (time_max_vals - time_min_global) / (time_max_global - time_min_global + 1e-8)
    phase_min_norm = (phase_min_vals - phase_min_global) / (phase_max_global - phase_min_global + 1e-8)
    phase_max_norm = (phase_max_vals - phase_min_global) / (phase_max_global - phase_min_global + 1e-8)

    # map to pixel
    tmin_px = (time_min_norm * (img_size - 1)).astype(np.int32)
    tmax_px = (time_max_norm * (img_size - 1)).astype(np.int32)
    pmin_px = (phase_min_norm * (img_size - 1)).astype(np.int32)
    pmax_px = (phase_max_norm * (img_size - 1)).astype(np.int32)

    # draw box
    for trial in range(trials):
        for w in range(windows):
            x1, x2 = tmin_px[trial, w], tmax_px[trial, w]
            y1, y2 = pmin_px[trial, w], pmax_px[trial, w]
            # phase is y ax, need to clip
            y1_img = img_size - 1 - y2
            y2_img = img_size - 1 - y1
            # avoid cross border
            x1, x2 = np.clip([x1, x2], 0, img_size - 1)
            y1_img, y2_img = np.clip([y1_img, y2_img], 0, img_size - 1)
            # fill
            # output[trial, y1_img:y2_img+1, x1:x2+1] = 1
            # Only four lines around the border
            output[trial, y1_img, x1:x2 + 1] = 1
            output[trial, y2_img, x1:x2 + 1] = 1
            output[trial, y1_img:y2_img + 1, x1] = 1
            output[trial, y1_img:y2_img + 1, x2] = 1

    return output


c1_R = Gr2Rgb(c1_R_t, c1_R_p)
print(c1_R.shape)
c1_G = Gr2Rgb(c1_G_t, c1_G_p)
print(c1_G.shape)
c1_B = Gr2Rgb(c1_B_t, c1_B_p)
print(c1_B.shape)

c2_R = Gr2Rgb(c2_R_t, c2_R_p)
print(c2_R.shape)
c2_G = Gr2Rgb(c2_G_t, c2_G_p)
print(c2_G.shape)
c2_B = Gr2Rgb(c2_B_t, c2_B_p)
print(c2_B.shape)



c1_rgb = np.stack([c1_R, c1_G, c1_B], axis=-1)
c2_rgb = np.stack([c2_R, c2_G, c2_B], axis=-1)

print(c1_rgb.shape)
print(c2_rgb.shape)

np.save('c1_rgb.npy', c1_rgb)
np.save('c2_rgb.npy', c2_rgb)

