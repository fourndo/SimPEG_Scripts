# -*- coding: utf-8 -*-
"""
Created on Wed May  9 13:20:56 2018

@author: fourndo@gmail.com


Run an equivalent source inversion

"""
from SimPEG import (
    Mesh, Utils, Maps, Regularization, Regularization,
    DataMisfit, Inversion, InvProblem, Directives, Optimization,
    )
import SimPEG.PF as PF
import numpy as np
import os
import json
from discretize.utils import meshutils
from scipy.spatial import Delaunay
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import cKDTree
from SimPEG.Utils import mkvc
import dask
from dask.distributed import Client
import multiprocessing
import sys


input_file = sys.argv[1]

if input_file is not None:
    workDir = os.path.sep.join(
                input_file.split(os.path.sep)[:-1]
            )
    if len(workDir) > 0:
        workDir += os.path.sep

    # self.readDriverFile(input_file.split(os.path.sep)[-1])

else:

    assert input_file is not None, "The input file is missing: 'python PFinversion.py input_file.json'"

# workDir = ".\\Assets\\MAG"
outDir = workDir + "SimPEG_PFInversion" + os.path.sep


# Default parameter values
parallelized = 'dask'
meshType = 'TREE'
tileProblem = True
mesh = None
topo = False
ndv = -100

dsep = os.path.sep

os.system('mkdir ' + outDir)

# Read json file and overwrite defaults
with open(input_file, 'r') as f:
    driver = json.load(f)

input_dict = dict((k.lower(), driver[k]) for k in list(driver.keys()))

# Deal with the data
if input_dict["forward_type"].lower() == 'grav':

    survey = Utils.io_utils.readUBCgravityObservations(workDir + input_dict["data_file"])

elif input_dict["forward_type"].lower() in ['mag', 'mvi']:

    survey, H0 = Utils.io_utils.readUBCmagneticsObservations(workDir + input_dict["data_file"])

else:
    assert False, "PF Inversion only implemented for 'dataFile' 'grav' | 'mag' | 'mvi' "

print(survey)
if "mesh_file" in list(driver.keys()):
    mesh = Mesh.TreeMesh.readUBC(workDir + input_dict["mesh_file"])

if "model_file" in list(driver.keys()):
    model = mesh.readModelUBC(workDir + input_dict["model_file"])


rxLoc = survey.srcField.rxList[0].locs

# Going through all problems:
# 1- Pair the survey and problem
# 2- Add up sensitivity weights
# 3- Add to the ComboMisfit

# Create first mesh outside the parallel process


# Compute active cells
activeCells = model != input_dict["no_data_value"]
nC = int(activeCells.sum())  # Number of active cells
print(nC)
idenMap = Maps.IdentityMap(nP=nC)
# Get the layer of cells directly below topo


if input_dict["forward_type"].lower() == 'grav':
    indenMap = Maps.IdentityMap(nP=nC)
    prob = PF.Gravity.GravityIntegral(
        mesh, rhoMap=idenMap, actInd=activeCells, forwardOnly=True,
        )

elif input_dict["forward_type"].lower() == 'mag':
    indenMap = Maps.IdentityMap(nP=nC)
    prob = PF.Magnetics.MagneticIntegral(
        mesh, chiMap=idenMap, actInd=activeCells, forwardOnly=True,
        )

elif input_dict["forward_type"].lower() == 'mvi':
    indenMap = Maps.IdentityMap(nP=3*nC)
    prob = PF.Magnetics.MagneticIntegral(
        mesh, chiMap=idenMap, actInd=activeCells,
        modelType='vector', forwardOnly=True
    )

prob.pair(survey)
dpred = prob.fields(model[activeCells])



if input_dict["forward_type"].lower() == 'grav':

    Utils.io_utils.writeUBCgravityObservations(outDir + 'Predicted.dat', survey, dpred)

elif input_dict["forward_type"].lower() in ['mvi', 'mag']:

    Utils.io_utils.writeUBCmagneticsObservations(outDir + 'Predicted.dat', survey, dpred)


# if "forward" in list(driver.keys()):
#     if input_dict["forward"][0] == "DRAPE":
#         print("DRAPED")
#         # Define an octree mesh based on the data
#         nx = int((rxLoc[:, 0].max()-rxLoc[:, 0].min()) / input_dict["forward"][1])
#         ny = int((rxLoc[:, 1].max()-rxLoc[:, 1].min()) / input_dict["forward"][2])
#         vectorX = np.linspace(rxLoc[:, 0].min(), rxLoc[:, 0].max(), nx)
#         vectorY = np.linspace(rxLoc[:, 1].min(), rxLoc[:, 1].max(), ny)

#         x, y = np.meshgrid(vectorX, vectorY)

#         # Only keep points within max distance
#         tree = cKDTree(np.c_[rxLoc[:, 0], rxLoc[:, 1]])
#         # xi = _ndim_coords_from_arrays(, ndim=2)
#         dists, indexes = tree.query(np.c_[mkvc(x), mkvc(y)])

#         x = mkvc(x)[dists < input_dict["forward"][4]]
#         y = mkvc(y)[dists < input_dict["forward"][4]]

#         z = F(mkvc(x), mkvc(y)) + input_dict["forward"][3]
#         newLocs = np.c_[mkvc(x), mkvc(y), mkvc(z)]

#     elif input_dict["forward"][0] == "UpwardContinuation":
#         newLocs = rxLoc.copy()
#         newLocs[:, 2] += input_dict["forward"][1]

#     if input_dict["forward_type"].lower() == 'grav':
#         rxLoc = PF.BaseGrav.RxObs(newLocs)
#         srcField = PF.BaseGrav.SrcField([rxLoc])
#         forward = PF.BaseGrav.LinearSurvey(srcField)

#     elif input_dict["forward_type"].lower() == 'mag':
#         rxLoc = PF.BaseMag.RxObs(newLocs)
#         srcField = PF.BaseMag.SrcField([rxLoc], param=survey.srcField.param)
#         forward = PF.BaseMag.LinearSurvey(srcField)

#     forward.std = np.ones(newLocs.shape[0])

#     activeGlobal = (activeCellsMap * invProb.model) != ndv
#     idenMap = Maps.IdentityMap(nP=int(activeGlobal.sum()))

#     if input_dict["forward_type"].lower() == 'grav':
#         fwrProb = PF.Gravity.GravityIntegral(
#             mesh, rhoMap=idenMap, actInd=activeCells,
#             n_cpu=n_cpu, forwardOnly=True, rxType='xyz'
#             )
#     elif input_dict["forward_type"].lower() == 'mag':
#         fwrProb = PF.Magnetics.MagneticIntegral(
#             mesh, chiMap=idenMap, actInd=activeCells,
#             n_cpu=n_cpu, forwardOnly=True, rxType='xyz'
#             )

#     forward.pair(fwrProb)
#     pred = fwrProb.fields(invProb.model)

#     if input_dict["forward_type"].lower() == 'grav':

#         Utils.io_utils.writeUBCgravityObservations(outDir + 'Forward.dat', forward, pred)

#     elif input_dict["forward_type"].lower() == 'mag':

#         Utils.io_utils.writeUBCmagneticsObservations(outDir + 'Forward.dat', forward, pred)
