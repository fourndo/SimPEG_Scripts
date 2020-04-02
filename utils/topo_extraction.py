from os.path import isfile, join
from os import listdir, sep
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import xarray as xr


def get_topo(center, width, height, resolution=None, angle=0, repository="."):
    zarr_files = [f for f in listdir(repository) if "zarr" in f]

    ew_rot = rotation(np.c_[center[0], center[1]], center, angle)

    data_lim_x = [ew_rot[0, 0] - width / 2, ew_rot[0, 0] + width / 2]
    data_lim_y = [ew_rot[0, 1] - height / 2, ew_rot[0, 1] + height / 2]

    dtm = []

    for file in zarr_files:
        topo = xr.open_zarr(repository + sep + file)
        xx = topo['x'].values
        yy = topo['y'].values

        if not (
            any(np.searchsorted(xx, data_lim_x)) and
            any(np.searchsorted(yy, data_lim_y))
        ):
            continue

        if resolution is not None:
            ii = int(resolution / np.min(xx[1:] - xx[:-1]))
        else:
            ii = 1

        X, Y = np.meshgrid(xx[::ii], yy[::ii])
        xx = X.flatten()
        yy = Y.flatten()

        if angle != 0:
            xy = rotation(np.c_[xx, yy], center, angle)
        else:
            xy = np.c_[xx, yy]

        ind = (
                (xy[:, 0] > data_lim_x[0] - width / 2) *
                (xy[:, 0] < data_lim_x[1] + width / 2) *
                (xy[:, 1] > data_lim_y[0] - height / 2) *
                (xy[:, 1] < data_lim_y[1] + height / 2)
        )

        if np.any(ind):
            d = np.asarray(topo['topo'][::ii, ::ii])
            d = d.flatten()[ind]

            temp = np.c_[xy[ind, :], d]
            ind = d > -999

            dtm.append(temp[ind, :])

    dtm = np.vstack(dtm)

    # Rotate back to local
    if angle != 0:
        dtm = rotation(dtm, center, -angle)

    return dtm


def rotation(xyz, center, angle):
    rotation = np.r_[
        np.c_[np.cos(np.pi * angle / 180), -np.sin(np.pi * angle / 180)],
        np.c_[np.sin(np.pi * angle / 180), np.cos(np.pi * angle / 180)]
    ]

    xyz[:, 0] -= center[0]
    xyz[:, 1] -= center[1]

    xy_rot = np.dot(rotation, xyz[:, :2].T).T

    return np.c_[xy_rot[:, 0] + center[0], xy_rot[:, 1] + center[1], xyz[:, 2:]]