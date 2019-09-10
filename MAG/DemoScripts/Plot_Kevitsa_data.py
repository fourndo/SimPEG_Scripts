# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 12:39:00 2017

@author: DominiqueFournier
"""

from SimPEG import PF
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

work_dir = 'C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Kevitsa\\Modeling\\MAG\\Aiborne\\'

# Load data
survey = PF.Magnetics.readMagneticsObservations(work_dir + 'VTEM_FLT20m_IGRF53180nT.dat')


# Plot data
# fig = PF.Magnetics.plot_obs_2D(survey.srcField.rxList[0].locs, survey.dobs)

rxLoc = survey.srcField.rxList[0].locs
d = survey.dobs

vmin = -500
vmax = 2000
d[d>vmax] = vmax*.99

# Create grid of points
x = np.linspace(rxLoc[:, 0].min(), rxLoc[:, 0].max(), 100)
y = np.linspace(rxLoc[:, 1].min(), rxLoc[:, 1].max(), 100)

X, Y = np.meshgrid(x, y)

# Interpolate
d_grid = griddata(rxLoc[:, 0:2], d, (X, Y), method='linear')

fig = plt.figure()
axs = plt.subplot()


# plt.imshow(d_grid, extent=[x.min(), x.max(), y.min(), y.max()],
#            origin='lower', vmin=vmin, vmax=vmax, cmap="plasma")
plt.contourf(X, Y, d_grid, 30, vmin=vmin, vmax=vmax, cmap="viridis")
plt.colorbar(fraction=0.02)
plt.contour(X, Y, d_grid, 10,colors='k', vmin=vmin, vmax=vmax, linewidths=1)

axs.

plt.show()
