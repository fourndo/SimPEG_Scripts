# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 09:47:38 2016


@author: dominiquef
"""

from SimPEG import Mesh, Utils, np, PF, Maps, Problem, Survey, mkvc
import SimPEG.EM.FDEM as FDEM
import matplotlib.pyplot as plt
from SimPEG import SolverBiCG as Solver
from pymatsolver import PardisoSolver

# Define inducing field and sphere parameters
H0 = (50000., 60., 270.)
rad = 2.
rho = 0.1

# Define a mesh
cs = 0.5
hxind = [(cs,5,-1.3),(cs, 9),(cs,5,1.3)]
hyind = [(cs,5,-1.3),(cs, 9),(cs,5,1.3)]
hzind = [(cs,5,-1.3),(cs, 9),(cs,5,1.3)]
mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCC')

# Get cells inside the sphere
sph_ind = PF.MagAnalytics.spheremodel(mesh, 0., 0., 0., rad)

# Adjust susceptibility for volume difference
Vratio = (4./3.*np.pi*rad**3.) / (np.sum(sph_ind)*cs**3.)
model = np.ones(mesh.nC)*1e-8
model[sph_ind] = 0.01

rxLoc = np.asarray([np.r_[0,0, 4.]])

bzi = FDEM.Rx.Point_bSecondary(rxLoc, 'z', 'real')
bzr = FDEM.Rx.Point_bSecondary(rxLoc, 'z', 'imag')

freqs = [400]#np.logspace(2, 3, 5)
srcLoc = np.r_[0,0, 4.]

srcList = [
    FDEM.Src.MagDipole([bzr, bzi], freq, srcLoc, orientation='Z')
    for freq in freqs
]

mapping = Maps.IdentityMap(mesh)
surveyFD = FDEM.Survey(srcList)
prbFD = FDEM.Problem3D_b(mesh, sigmaMap=mapping, Solver=PardisoSolver)
prbFD.pair(surveyFD)
std = 0.03
surveyFD.makeSyntheticData(model, std)

#Mesh.TensorMesh.writeUBC(mesh,'MeshGrav.msh')
#Mesh.TensorMesh.writeModelUBC(mesh,'MeshGrav.den',model)
#PF.Gravity.writeUBCobs("Obs.grv",survey,d)