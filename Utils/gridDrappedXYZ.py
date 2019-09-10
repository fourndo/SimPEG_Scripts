from SimPEG.Utils import mkvc
import scipy as sp
import numpy as np
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:02:31 2017

@author: DominiqueFournier
"""

def gridDrappedXYZ(width, height, spacing, x0=(0, 0, 0), topo=None):
    """
        grid_survey(width)
        Generate concentric grid surveys centered at the origin
        :grid: List of grid spacing
        :width: Grid width

        return: rxLoc an n-by-3 receiver locations

    """

    nCx = int(width/spacing)
    nCy = int(height/spacing)
    
    dx = width/nCx
    dy = height/nCy
    
    rxVec = -width/2. + dx/2. + np.asarray(range(nCx))*dx
    ryVec = -height/2. + dy/2. + np.asarray(range(nCy))*dy
    
    rxGridx, rxGridy = np.meshgrid(rxVec, ryVec)

    rxGridx += x0[0]
    rxGridy += x0[1]

    if topo is not None:

        rxGridz = sp.interpolate.griddata(topo[:, :2], topo[:, 2],
                                             (rxGridx, rxGridy),
                                             method='linear') + x0[2]
    else:
        rxGridz = np.zeros_like(rxGridx) + x0[2]
            
    
    return np.c_[mkvc(rxGridx), mkvc(rxGridy), mkvc(rxGridz)]

# Run it like this
xyz = gridDrappedXYZ(10., 10., 2.)