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
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
from SimPEG.Utils import mkvc
from dask.distributed import Client


input_file = r"C:\Users\DominiqueFournier\Dropbox\Projects\Synthetic\Block_Gaussian_topo\PF_forward_input.json"#sys.argv[1]
dsep = os.path.sep

grav_components = ['g_z']
mag_components = [
        "dbx_dx", "dbx_dy", "dbx_dz", "dby_dy",
        "dby_dz", "dbz_dz", "bx", "by", "bz", "tmi"
        ]

assert input_file is not None, (
        "The input file is missing. Run this script with: " +
        "'python PFinversion.py input_file.json'"
        )

workDir = os.path.sep.join(
            input_file.split(os.path.sep)[:-1]
        )
if len(workDir) > 0:
    workDir += os.path.sep

# Read json file
with open(input_file, 'r') as f:
    driver = json.load(f)

# Convert keys to lower by default
input_dict = dict((k.lower(), driver[k]) for k in list(driver.keys()))

components = input_dict["data_type"]
if not isinstance(components, list):
    components = [components]
    
n_components = len(components)
is_mag_simulation = False
for component in components:
    
    assert component in grav_components + mag_components, (
        "Input data_type '{}' not implemented. Please choose from {:}".format(component, 
            grav_components + mag_components
        )
    )
    
    if component in mag_components:
        is_mag_simulation = True

    if (component in grav_components) and (is_mag_simulation):
        
        assert False, "Cannot simulate 'mag' or 'grav' components together"

if "mesh_file" in list(driver.keys()):

    # Detect if a Tree or Tensor mesh file
    fid = open(workDir + input_dict["mesh_file"], 'r')
    for ii in range(6):
        line = fid.readline()
    fid.close()

    if line:
        mesh = Mesh.TreeMesh.readUBC(workDir + input_dict["mesh_file"])
    else:
        mesh = Mesh.TensorMesh.readUBC(workDir + input_dict["mesh_file"])

if "model_file" in list(driver.keys()):

    fid = open(workDir + input_dict["model_file"], 'r')
    line = fid.readline()
    fid.close()

    if np.array(line.split(), dtype=float).shape[0] > 1:
        model_type = 'vector'
        model = Utils.io_utils.readVectorUBC(
            mesh, workDir + input_dict["model_file"]
        )
    else:
        model_type = 'scalar'
        model = mesh.readModelUBC(workDir + input_dict["model_file"])


obsFile = input_dict["data_locations"]
if not is_mag_simulation:

    survey = Utils.io_utils.readUBCgravityObservations(workDir + obsFile)

else:

    survey, H0 = Utils.io_utils.readUBCmagneticsObservations(workDir + obsFile)

# Deal with the data
if "draped_grid" in list(driver.keys()):

    rxLoc = survey.rxLoc

    # Get grid extent
    nx = int(
        (rxLoc[:, 0].max()-rxLoc[:, 0].min()) /
        input_dict["draped_grid"][0]
    )
    ny = int(
        (rxLoc[:, 1].max()-rxLoc[:, 1].min()) /
        input_dict["draped_grid"][1]
    )
    vectorX = np.linspace(rxLoc[:, 0].min(), rxLoc[:, 0].max(), nx)
    vectorY = np.linspace(rxLoc[:, 1].min(), rxLoc[:, 1].max(), ny)

    x, y = np.meshgrid(vectorX, vectorY)

    # Only keep points within max distance
    tree = cKDTree(np.c_[rxLoc[:, 0], rxLoc[:, 1]])
    # xi = _ndim_coords_from_arrays(, ndim=2)
    dists, indexes = tree.query(np.c_[mkvc(x), mkvc(y)])

    x = mkvc(x)[dists < input_dict["draped_grid"][3]]
    y = mkvc(y)[dists < input_dict["draped_grid"][3]]

    # Load topo file
    topo = np.genfromtxt(
        workDir + input_dict["draped_grid"][2], skip_header=1
    )
    # Compute delaunay triangulation
    tri2D = Delaunay(topo[:, :2])
    F = LinearNDInterpolator(tri2D, topo[:, 2])
    z = F(rxLoc[:, :2])

    newLocs = np.c_[mkvc(x), mkvc(y), mkvc(z)]

    # Create survey
    if not is_mag_simulation:
        rxLoc = PF.BaseGrav.RxObs(newLocs)
        srcField = PF.BaseGrav.SrcField([rxLoc])
        survey = PF.BaseGrav.LinearSurvey(srcField)

    else:
        rxLoc = PF.BaseMag.RxObs(newLocs)
        srcField = PF.BaseMag.SrcField([rxLoc], param=survey.srcField.param)
        survey = PF.BaseMag.LinearSurvey(srcField)

# If upward_continuation distance is provided
if "upward_continuation" in list(driver.keys()):
    survey.srcField.rxList[0].locs[:, 2] += input_dict["upward_continuation"]

# Assign components to be calculated
survey.components = components

# Compute active cells
if model_type == 'scalar':
    activeCells = model != input_dict["no_data_value"]
    m0 = model[activeCells]
else:
    activeCells = model[:,0] != input_dict["no_data_value"]
    m0 = mkvc(model[activeCells, :])

nC = int(activeCells.sum())  # Number of active cells

# Get the layer of cells directly below topo
if not is_mag_simulation:
    idenMap = Maps.IdentityMap(nP=nC)
    prob = PF.Gravity.GravityIntegral(
        mesh, rhoMap=idenMap, actInd=activeCells, forwardOnly=True,
        )

else:

    if model_type == 'scalar':
        idenMap = Maps.IdentityMap(nP=nC)
        prob = PF.Magnetics.MagneticIntegral(
            mesh, chiMap=idenMap, actInd=activeCells, forwardOnly=True,
            )

    else:
        idenMap = Maps.IdentityMap(nP=3*nC)
        prob = PF.Magnetics.MagneticIntegral(
            mesh, chiMap=idenMap, actInd=activeCells,
            modelType='vector', forwardOnly=True
        )

prob.pair(survey)
dpred = prob.fields(m0)


for ii, component in enumerate(components):
    
    file_name = 'Predicted_' + component +'.dat'
    if component in grav_components:
    
        Utils.io_utils.writeUBCgravityObservations(workDir + file_name, survey, dpred[ii::n_components])
    
    elif component in mag_components:
    
        Utils.io_utils.writeUBCmagneticsObservations(workDir + file_name, survey, dpred[ii::n_components])
