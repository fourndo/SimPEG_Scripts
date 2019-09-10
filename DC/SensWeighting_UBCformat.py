# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 14:21:35 2017

@author: DominiqueFournier
"""

from SimPEG import Mesh, Utils, EM, Maps, Survey
from SimPEG import DataMisfit, Regularization, Optimization
from SimPEG import Directives, InvProblem, Inversion
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from SimPEG.EM.Static import DC, IP
from pymatsolver import PardisoSolver

mesh = Mesh.TensorMesh.readUBC('MeshDC.msh')

jUBC = mesh.readModelUBC('UBC_DCIP3D\\sensitivity.txt')

j = jUBC**0.5

j /= np.max(j)

jface = mesh.aveCC2F * j

jFx = jface[:mesh.nFx].reshape((mesh.nCx+1,mesh.nCy,mesh.nCz), order='F')
jFxT = jFx.transpose((2, 0, 1))
# Flip z to positive down
jFxTR = Utils.mkvc(jFxT[::-1, :, :])**0
        
jFy = jface[mesh.nFx:mesh.nFx+mesh.nFy].reshape((mesh.nCx,mesh.nCy+1,mesh.nCz), order='F')
jFyT = jFy.transpose((2, 0, 1))
# Flip z to positive down
jFyTR = Utils.mkvc(jFyT[::-1, :, :])**0

jFz = jface[mesh.nFx+mesh.nFy:].reshape((mesh.nCx,mesh.nCy,mesh.nCz+1), order='F')
jFzT = jFz.transpose((2, 0, 1))
# Flip z to positive down
jFzTR = Utils.mkvc(jFzT[::-1, :, :])**0

normj = np.linalg.norm(j)
jw = np.r_[j, jFxTR, jFyTR, jFzTR]

np.savetxt('UBC_DCIP3D\\wmat.dat',jw, fmt='%.6e')


