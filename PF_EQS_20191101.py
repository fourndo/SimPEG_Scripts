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
import matplotlib.pyplot as plt
import os
import json
from discretize.utils import meshutils
#from scipy.spatial import Delaunay
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from scipy.spatial import cKDTree
from SimPEG.Utils import mkvc
import dask
# from dask.distributed import Client
import multiprocessing
import sys
import time
import geosoft.gxpy.grid as gxgrid
import geosoft.gxpy.gx as gx

# NEED TO ADD ALPHA VALUES
# NEED TO ADD REFERENCE
# NEED TO ADD STARTING

###############################################################################
# EQS Setup
def drapeOverTopgraphy(survey, topo, survey_altitude):
    print('Interpolating obs locations...')

    rxLoc = survey.srcField.rxList[0].locs
    max_pad_distance = max([5 * survey_altitude, 500]) # Only used to minimize the extent of topo resampling for large problems
    
    # Create new data locations draped at drapeAltitude above topo
    ix = (topo[:, 0] >= (rxLoc[:, 0].min() - max_pad_distance)) & (topo[:, 0] <= (rxLoc[:, 0].max() + max_pad_distance)) & \
         (topo[:, 1] >= (rxLoc[:, 1].min() - max_pad_distance)) & (topo[:, 1] <= (rxLoc[:, 1].max() + max_pad_distance))

    F = LinearNDInterpolator(topo[ix, :2], topo[ix, 2])

    oldObs = survey.dobs # copy old st. devs.
    oldStd = survey.std # copy old st. devs.
    newRx = np.c_[rxLoc[:, :2], F(rxLoc[:,:2]) + survey_altitude]
    rxLocNew = PF.BaseMag.RxObs(newRx)
    srcField = PF.BaseMag.SrcField([rxLocNew], param=survey.srcField.param)
    survey = PF.BaseMag.LinearSurvey(srcField)
    survey.dobs = oldObs
    survey.std = oldStd

    fig, ax = plt.subplots(1,2,sharey=True,constrained_layout=True)
    ax[0].scatter(rxLoc[:,0], rxLoc[:,1], s=0.1, marker='.',c=rxLoc[:,2])
    ax[0].set_aspect('equal')
    ax[0].set_title('Original elev')
    ax[1].scatter(newRx[:,0], newRx[:,1], s=0.1, marker='.',c=newRx[:,2])
    ax[1].set_aspect('equal')
    ax[1].set_title('Draped elev')
    plt.show(block=False)

    fig, ax = plt.subplots(1,2,sharey=True,constrained_layout=True)
    ax[0].scatter(rxLoc[:,0], rxLoc[:,1], s=0.1, marker='.',c=oldStd)
    ax[0].set_aspect('equal')
    ax[0].set_title('Original std')
    ax[1].scatter(newRx[:,0], newRx[:,1], s=0.1, marker='.',c=survey.std)
    ax[1].set_aspect('equal')
    ax[1].set_title('Draped std')
    plt.show(block=False)

    return survey

def readGeosoftGrid(gridname, topo, survey_altitude, data_type, inducing_field_aid=None):
    geosoft_input = True
    geosoft_output = True
    
    try:
        # open the grid
        with gxgrid.Grid.open(gridname) as grid:
    
            # get the data in a numpy array
            GsftGrid_values = grid.xyzv()
            # GsftGrid_mask = np.ma.masked_invalid(GsftGrid_values[:,:,3])
            GsftGrid_mask = ~np.isnan(GsftGrid_values)
            # print(GsftGrid_mask)
            GsftGrid_nx = grid.nx
            GsftGrid_ny = grid.ny
            GsftGrid_props = grid.properties()
    
        newLocs = np.reshape(GsftGrid_values[:, :, :3],(GsftGrid_nx * GsftGrid_ny, 3))
        newLocs_mask = ~np.isnan(np.reshape(GsftGrid_values[:, :, 3],(GsftGrid_nx * GsftGrid_ny, 1)))
        newLocs = newLocs[newLocs_mask.squeeze(),:]
    
        # Create forward obs elevations
        # Create new data locations draped at survey_altitude above topo
        ix = (topo[:, 0] >= (newLocs[:, 0].min() - max_distance)) & (topo[:, 0] <= (newLocs[:, 0].max() + max_distance)) & \
             (topo[:, 1] >= (newLocs[:, 1].min() - max_distance)) & (topo[:, 1] <= (newLocs[:, 1].max() + max_distance))
        F = LinearNDInterpolator(topo[ix, :2], topo[ix, 2] + survey_altitude)

        newLocs[:,2] = F(newLocs[:,:2])
        if data_type == 'geosoft_mag':
            # Mag only
            rxLocNew = PF.BaseMag.RxObs(newLocs)
            # retain TF, but update inc-dec to vertical field
            srcField = PF.BaseMag.SrcField([rxLocNew], param=inducing_field_aid)
            survey = PF.BaseMag.LinearSurvey(srcField, components=['tmi'])
            survey.dobs = newLocs[:,2]
        else:
            # Grav only
            rxLoc = PF.BaseGrav.RxObs(newLocs)
            srcField = PF.BaseGrav.SrcField([rxLoc])
            survey = PF.BaseGrav.LinearSurvey(srcField)
            survey.dobs = -newLocs[:,2]
            
        fig, ax1 = plt.figure(), plt.subplot()
        plt.scatter(newLocs[:,0], newLocs[:,1], s=0.01, marker='.',c=newLocs[:,2])
        ax1.set_aspect('equal')
        plt.show(block=False)
    
        print('Will forward model to Geosoft Grid...')
    except:
        geosoft_input = False
        geosoft_output = False
        GsftGrid_mask = None
        GsftGrid_values = None
        GsftGrid_props = None
                    
        newLocs = rxLoc.copy()
        print('Will forward model to UBC-GIF Format...')

    return survey, geosoft_output, geosoft_input, GsftGrid_values, GsftGrid_mask, GsftGrid_props

geosoft_enabled = False
geosoft_output = False
geosoft_input = False
try:
    gxc = gx.GXpy()
    geosoft_enabled = True
except:
    print("Geosoft was not found.")

dsep = os.path.sep
input_file = sys.argv[1]

if input_file is not None:
    workDir = dsep.join(
                input_file.split(dsep)[:-1]
            )
    if len(workDir) > 0:
        workDir += dsep
    else:
        workDir = os.getcwd() + dsep
    print('Working directory:', workDir)
    
else:

    assert input_file is not None, "The input file is missing: 'python PFinversion.py input_file.json'"

# Read json file and overwrite defaults
with open(input_file, 'r') as f:
    driver = json.load(f)

input_dict = dict((k.lower(), driver[k]) for k in list(driver.keys()))

assert "inversion_type" in list(input_dict.keys()), "Require 'inversion_type' to be set: 'grav', 'mag', 'mvi', or 'mvs'"
assert input_dict["inversion_type"].lower() in ['grav', 'mag', 'mvi', 'mvs'], "'inversion_type' must be one of: 'grav', 'mag', 'mvi', or 'mvs'"

if "result_folder" in list(input_dict.keys()):
    outDir = workDir + dsep + input_dict["result_folder"] + dsep
else:
    outDir = workDir + dsep + "SimPEG_PFInversion" + dsep


# Default parameter values
parallelized = 'dask'
meshType = 'TREE'

ndv = -100

os.system('mkdir ' + outDir)

###############################################################################
# Deal with the EQS Settings
if "eqs_mvi" in list(input_dict.keys()):
    eqs_mvi = input_dict["eqs_mvi"]
else:
    eqs_mvi = False
    
if "topography" in list(input_dict.keys()):
    topo = np.genfromtxt(workDir + input_dict["topography"],
                         skip_header=1)
#        topo[:,2] += 1e-8
    # Compute distance of obs above topo
    F = NearestNDInterpolator(topo[:, :2], topo[:, 2])
elif eqs_mvi:
    print('EQS needs topography')
    quit()
else:
    topo = None
    
if "survey_altitude" in list(input_dict.keys()):
    survey_altitude = input_dict["survey_altitude"]
else:
    survey_altitude = 100

if "upward_continue" in list(input_dict.keys()):
    upward_continue = input_dict["upward_continue"]
else:
    upward_continue = 0.0

###############################################################################
# Deal with the data
if "inducing_field_aid" in list(input_dict.keys()):
    inducing_field_aid = input_dict["inducing_field_aid"]
    
    assert (len(inducing_field_aid) == 3 and inducing_field_aid[0] > 0)
else:
    inducing_field_aid = None

if input_dict["data_type"].lower() in ['ubc_grav']:

    survey = Utils.io_utils.readUBCgravityObservations(workDir + input_dict["data_file"])

elif input_dict["data_type"].lower() in ['ubc_mag']:

    survey, H0 = Utils.io_utils.readUBCmagneticsObservations(workDir + input_dict["data_file"])
    survey.components = ['tmi']

elif input_dict["data_type"].lower() in ['ftmg']:

    patch = Utils.io_utils.readFTMGdataFile(file_path=workDir + input_dict["data_file"])
    component = input_dict["ftmg_components"]
    H0 = np.asarray(input_dict["inducing_field_aid"])

    if "limits" in list(input_dict.keys()):
        limits = input_dict["limits"]
    else:
        limits = None

    xs, ys, zs = patch.getUtmLocations(limits=limits, ground=True)
    survey = patch.createFtmgSurvey(inducing_field=H0, force_comp=component)

    print(survey.components, component, survey.nD)

elif geosoft_enabled and input_dict["data_type"].lower() in ['geosoft_mag', 'geosoft_grav']:

    assert inducing_field_aid, "Geosoft input needs requires 'inducing_field_aid' entry"
    assert "new_uncert" in list(input_dict.keys()), "Geosoft input needs requires 'new_uncert' entry"
    
    survey, geosoft_output, geosoft_input, GsftGrid_values, GsftGrid_mask, GsftGrid_props = readGeosoftGrid(input_dict["data_file"], topo, survey_altitude, input_dict["data_type"].lower(), inducing_field_aid)
        
else:

    assert False, "PF EQS only implemented for 'data_type' 'ubc_grav' | 'ubc_mag' | 'geosoft_grav' | 'geosoft_mag' "

# 0-level the data if required, d0 = 0 level
d0 = 0.0
if "subtract_mean" in list(input_dict.keys()) and input_dict["data_type"].lower() in ['ubc_mag', 'ubc_grav']:
    subtract_mean = input_dict["subtract_mean"]
    if subtract_mean:
        d0 = np.mean(survey.dobs)
        survey.dobs -= d0
        print('Removed data mean: {0:.6g}'.format(d0))
else:
    subtract_mean = False

# Update the specified data uncertainty
if "new_uncert" in list(input_dict.keys()) and input_dict["data_type"].lower() in ['ubc_mag', 'ubc_grav']:
    new_uncert = input_dict["new_uncert"]
    if new_uncert:
        assert (len(new_uncert) == 2 and all(np.asarray(new_uncert) >= 0)), "New uncertainty requires pct fraction (0-1) and floor."
        survey.std = np.maximum(abs(new_uncert[0]*survey.dobs),new_uncert[1])
else:
    new_uncert = False
    
if survey.std is None:
    survey.std = survey.dobs * 0 + 1 # Default

print('Min uncert: {0:.6g} nT'.format(survey.std.min()))

if "drape_over_topo" in list(input_dict.keys()):
    drape_over_topo = input_dict["upward_continue"]
    survey = drapeOverTopgraphy(survey, topo, survey_altitude)
else:
    drape_over_topo = False
    
###############################################################################
# Manage other inputs
if "mesh_file" in list(input_dict.keys()):
    meshInput = Mesh.TreeMesh.readUBC(workDir + input_dict["mesh_file"])
else:
    meshInput = None

if topo is None:
    if "topography" in list(input_dict.keys()):
        topo = np.genfromtxt(workDir + input_dict["topography"],
                             skip_header=1)
    #        topo[:,2] += 1e-8
        # Compute distance of obs above topo
        F = NearestNDInterpolator(topo[:, :2], topo[:, 2])
    
    else:
        # Grab the top coordinate and make a flat topo
        indTop = meshInput.gridCC[:, 2] == meshInput.vectorCCz[-1]
        topo = meshInput.gridCC[indTop, :]
        topo[:, 2] += meshInput.hz.min()/2. + 1e-8
        F = NearestNDInterpolator(topo[:, :2], topo[:, 2])
    
if "target_chi" in list(input_dict.keys()):
    target_chi = input_dict["target_chi"]
else:
    target_chi = 1

if "model_norms" in list(input_dict.keys()):
    model_norms = input_dict["model_norms"]
else:
    model_norms = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

model_norms = np.c_[model_norms]

if input_dict["inversion_type"].lower() in ['grav', 'mag']:
    model_norms = model_norms[:4]
    assert model_norms.shape[0] == 4, "Model norms need at least for values (p_s, p_x, p_y, p_z)"
else:
    assert model_norms.shape[0] == 12, "Model norms needs 12 terms for [a, t, p] x [p_s, p_x, p_y, p_z]"
    
if "max_IRLS_iter" in list(input_dict.keys()):
    max_IRLS_iter = input_dict["max_IRLS_iter"]
    assert max_IRLS_iter >= 0, "Max IRLS iterations must be >= 0"
else:
    if (input_dict["inversion_type"].lower() != 'mvis') or (np.all(model_norms==2)):
        # Cartesian or not sparse
        max_IRLS_iter = 0
    else:
        # Spherical or sparse 
        max_IRLS_iter = 20    

if "gradient_type" in list(input_dict.keys()):
    gradient_type = input_dict["gradient_type"]
else:
    gradient_type = 'total'

if "n_cpu" in list(input_dict.keys()):
    n_cpu = input_dict["n_cpu"]
else:
    n_cpu = multiprocessing.cpu_count()

if "max_ram" in list(input_dict.keys()):
    max_ram = input_dict["max_ram"]
else:
    max_ram = 4

if "padding_distance" in list(input_dict.keys()):
    padding_distance = input_dict["padding_distance"]
else:
    padding_distance = [[0, 0], [0, 0], [0, 0]]

if "octree_levels_topo" in list(input_dict.keys()):
    octree_levels_topo = input_dict["octree_levels_topo"]
else:
    octree_levels_topo = [0, 1]

if "octree_levels_obs" in list(input_dict.keys()):
    octree_levels_obs = input_dict["octree_levels_obs"]
else:
    octree_levels_obs = [5, 5]

if "octree_levels_padding" in list(input_dict.keys()):
    octree_levels_padding = input_dict["octree_levels_padding"]
else:
    octree_levels_padding = [2, 2]

if "alphas" in list(input_dict.keys()):
    alphas = input_dict["alphas"]
else:
    alphas = [1, 1, 1, 1]

if "model_start" in list(input_dict.keys()):
    model_start = np.r_[input_dict["model_start"]]
    assert model_start.shape[0] == 1 or model_start.shape[0] == 3, "Start model needs to be a scalar or 3 component vector"
else:
    model_start = [0.0]

if "model_reference" in list(input_dict.keys()):
    model_reference = np.r_[input_dict["model_reference"]]
    assert model_reference.shape[0] == 1 or model_reference.shape[0] == 3, "Start model needs to be a scalar or 3 component vector"
else:
    model_reference = [0.0]

if len(octree_levels_padding) < len(octree_levels_obs):
    octree_levels_padding += octree_levels_obs[len(octree_levels_padding):]

if "core_cell_size" in list(input_dict.keys()):
    core_cell_size = input_dict["core_cell_size"]
else:
    assert("'core_cell_size' must be added to the inputs")

if "depth_core" in list(input_dict.keys()):
    depth_core = input_dict["depth_core"]
else:
    xLoc = survey.rxLoc[:, 0]
    yLoc = survey.rxLoc[:, 1]
    depth_core = np.max([
        (xLoc.max()-xLoc.min())/3, (yLoc.max()-yLoc.min())/3
    ])

if "max_distance" in list(input_dict.keys()):
    max_distance = input_dict["max_distance"]
else:
    max_distance = np.inf

if "tileProblem" in list(input_dict.keys()):
    tileProblem = input_dict["tileProblem"]
else:
    tileProblem = False

if "show_graphics" in list(input_dict.keys()):
    show_graphics = input_dict["show_graphics"]
else:
    show_graphics = False

if parallelized == 'dask':
    dask.config.set({'array.chunk-size': '256MiB'})
    dask.config.set(scheduler='threads')
    dask.config.set(num_workers=n_cpu)

###############################################################################
# Processing

rxLoc = survey.rxLoc

# Create near obs topo
newTopo = np.c_[rxLoc[:, :2], F(rxLoc[:, :2])]

# LOOP THROUGH TILES
surveyMask = np.ones(rxLoc.shape[0], dtype='bool')
# Going through all problems:
# 1- Pair the survey and problem
# 2- Add up sensitivity weights
# 3- Add to the ComboMisfit

# Create first mesh outside the parallel process

# Loop over different tile size and break problem until
# memory footprint false below max_ram
usedRAM = np.inf
count = 1
while usedRAM > max_ram:
    print("Tiling:" + str(count))

    if rxLoc.shape[0] > 40000:
        tiles, binCount, tileIDs, tile_numbers = Utils.modelutils.tileSurveyPoints(rxLoc, count, method='other')
    else:
        tiles, binCount, tileIDs, tile_numbers = Utils.modelutils.tileSurveyPoints(rxLoc, count)

    # Grab the smallest bin and generate a temporary mesh
    indMax = np.argmax(binCount)

    X1, Y1 = tiles[0][:, 0], tiles[0][:, 1]
    X2, Y2 = tiles[1][:, 0], tiles[1][:, 1]

    ind_t = tileIDs == tile_numbers[indMax]

    # Create the mesh and refine the same as the global mesh
    meshLocal = meshutils.mesh_builder_xyz(
        newTopo, core_cell_size, padding_distance=padding_distance, mesh_type='TREE', base_mesh=meshInput,
        depth_core=depth_core
    )

    if topo is not None:
        meshLocal = meshutils.refine_tree_xyz(
            meshLocal, topo, method='surface',
            octree_levels=octree_levels_topo, finalize=False
        )

    meshLocal = meshutils.refine_tree_xyz(
        meshLocal, rxLoc[ind_t, :], method='surface',
        max_distance=max_distance,
        octree_levels=octree_levels_obs,
        octree_levels_padding=octree_levels_padding,
        finalize=True,
    )


    tileLayer = Utils.surface2ind_topo(meshLocal, topo)

    # Calculate approximate problem size
    nDt, nCt = ind_t.sum()*1. * len(survey.components), tileLayer.sum()*1.

    nChunks = n_cpu  # Number of chunks
    cSa, cSb = int(nDt/nChunks), int(nCt/nChunks)  # Chunk sizes
    usedRAM = nDt * nCt * 8. * 1e-9
    count += 1
    print(nDt, nCt, usedRAM, binCount.min())
    del meshLocal
# After tiling:
# Plot data and tiles
#        fig, ax1 = plt.figure(), plt.subplot()
#        Utils.PlotUtils.plot2Ddata(rxLoc, survey.dobs, ax=ax1)
#        for ii in range(X1.shape[0]):
#            ax1.add_patch(Rectangle((X1[ii], Y1[ii]),
#                                 X2[ii]-X1[ii],
#                                 Y2[ii]-Y1[ii],
#                                 facecolor='none', edgecolor='k'))
#        ax1.set_xlim([X1.min()-20, X2.max()+20])
#        ax1.set_ylim([Y1.min()-20, Y2.max()+20])
#        ax1.set_aspect('equal')
#        plt.show()
nTiles = X1.shape[0]

def createLocalMesh(rxLoc, ind_t):

    # Remember selected data in case of tile overlap
    surveyMask[ind_t] = False

    # Create new survey
    if input_dict["inversion_type"].lower() == 'grav':
        rxLoc_t = PF.BaseGrav.RxObs(rxLoc[ind_t, :])
        srcField = PF.BaseGrav.SrcField([rxLoc_t])
        survey_t = PF.BaseGrav.LinearSurvey(srcField)
        survey_t.dobs = survey.dobs[ind_t]
        survey_t.std = survey.std[ind_t]
        survey_t.ind = ind_t

        # Utils.io_utils.writeUBCgravityObservations(outDir + "Tile" + str(ind) + '.dat', survey_t, survey_t.dobs)

    elif input_dict["inversion_type"].lower() in ['mag', 'mvi', 'mvis']:
        rxLoc_t = PF.BaseMag.RxObs(rxLoc[ind_t, :])
        srcField = PF.BaseMag.SrcField([rxLoc_t], param=survey.srcField.param)
        survey_t = PF.BaseMag.LinearSurvey(srcField, components=survey.components)

        dataInd = np.kron(ind_t, np.ones(len(survey.components))).astype('bool')

        survey_t.dobs = survey.dobs[dataInd]
        survey_t.std = survey.std[dataInd]
        survey_t.ind = ind_t

        # Utils.io_utils.writeUBCmagneticsObservations(outDir + "Tile" + str(ind) + '.dat', survey_t, survey_t.dobs)

    meshLocal = meshutils.mesh_builder_xyz(
        newTopo, core_cell_size, padding_distance=padding_distance, mesh_type='TREE', base_mesh=meshInput,
        depth_core=depth_core
    )

    if topo is not None:
        meshLocal = meshutils.refine_tree_xyz(
            meshLocal, topo, method='surface',
            octree_levels=octree_levels_topo, finalize=False
        )

    meshLocal = meshutils.refine_tree_xyz(
        meshLocal, rxLoc[ind_t, :], method='surface',
        max_distance=max_distance,
        octree_levels=octree_levels_obs,
        octree_levels_padding=octree_levels_padding,
        finalize=True,
    )

    # Create combo misfit function
    return meshLocal, survey_t


# Loop through the tiles and generate all sensitivities
print("Number of tiles:" + str(nTiles))
local_meshes, local_surveys = [], []
for tt in range(nTiles):

    print("Tile " + str(tt+1) + " of " + str(X1.shape[0]))

    local_mesh, local_survey = createLocalMesh(rxLoc, tileIDs==tile_numbers[tt])
    local_meshes += [local_mesh]
    local_surveys += [local_survey]

if meshInput is None:
    print("Creating Global Octree")
    mesh = meshutils.mesh_builder_xyz(
        newTopo, core_cell_size,
        padding_distance=padding_distance,
        mesh_type='TREE', base_mesh=meshInput,
        depth_core=depth_core
        )

    for local_mesh in local_meshes:

        mesh.insert_cells(local_mesh.gridCC, local_mesh.cell_levels_by_index(np.arange(local_mesh.nC)), finalize=False)

    mesh.finalize()
else:
    mesh = meshInput

if show_graphics:
    # Plot a slice
    slicePosition = rxLoc[:, 1].mean()
    
    sliceInd = int(round(np.searchsorted(mesh.vectorCCy, slicePosition)))
    fig, ax1 = plt.figure(), plt.subplot()
    im = mesh.plotSlice(np.log10(mesh.vol), normal='Y', ax=ax1, ind=sliceInd, grid=True, pcolorOpts={"cmap":"plasma"})
    ax1.set_aspect('equal')
    t = fig.get_size_inches()
    fig.set_size_inches(t[0]*2, t[1]*2)
    fig.savefig(outDir + 'Section_%.0f.png' % slicePosition, bbox_inches='tight', dpi=600)
    plt.show(block=False)

print("Calculating active cells from topo")
# Compute active cells
activeCells = Utils.surface2ind_topo(mesh, topo, gridLoc='CC')

Mesh.TreeMesh.writeUBC(
      mesh, outDir + 'OctreeMeshGlobal.msh',
      models={outDir + 'ActiveSurface.act': activeCells}
    )

if "adjust_clearance" in list(input_dict.keys()):

    print("Forming cKDTree for clearance calculations")
    tree = cKDTree(mesh.gridCC[activeCells, :])


# Get the layer of cells directly below topo
nC = int(activeCells.sum())  # Number of active cells

# Create active map to go from reduce set to full
activeCellsMap = Maps.InjectActiveCells(mesh, activeCells, ndv)

# Create identity map
if input_dict["inversion_type"].lower() in ['mvi', 'mvis']:
    wrGlobal = np.zeros(3*nC)
else:
    idenMap = Maps.IdentityMap(nP=nC)
    wrGlobal = np.zeros(nC)


def createLocalProb(meshLocal, survey_t, wrGlobal, ind):
    # createLocalProb(rxLoc, wrGlobal, lims, ind)
    # Generate a problem, calculate/store sensitivities for
    # given data points

    # Need to find a way to compute sensitivities only for intersecting cells
    activeCells_t = np.ones(meshLocal.nC, dtype='bool')  # meshUtils.modelutils.activeTopoLayer(meshLocal, topo)

    # Create reduced identity map
    if input_dict["inversion_type"].lower() in ['mvi', 'mvis']:
        nBlock = 3
    else:
        nBlock = 1

    tileMap = Maps.Tile((mesh, activeCells), (meshLocal, activeCells_t), nBlock=nBlock)

    activeCells_t = tileMap.activeLocal

    if "adjust_clearance" in list(input_dict.keys()):

        print("Setting Z values of data to respect clearance height")

        r, c_ind = tree.query(survey_t.rxLoc)
        dz = input_dict["adjust_clearance"]

        z = mesh.gridCC[activeCells, 2][c_ind] + mesh.h_gridded[activeCells, 2][c_ind]/2 + dz
        survey_t.srcField.rxList[0].locs[:, 2] = z

    if input_dict["inversion_type"].lower() == 'grav':

        Utils.io_utils.writeUBCgravityObservations(outDir + 'Survey_Tile' + str(ind) +'.dat', survey_t, survey_t.dobs)

    elif input_dict["inversion_type"].lower() == 'mag':

        Utils.io_utils.writeUBCmagneticsObservations(outDir + 'Survey_Tile' + str(ind) +'.dat', survey_t, survey_t.dobs)

    if input_dict["inversion_type"].lower() == 'grav':
        prob = PF.Gravity.GravityIntegral(
            meshLocal, rhoMap=tileMap, actInd=activeCells_t,
            parallelized=parallelized,
            Jpath=outDir + "Tile" + str(ind) + ".zarr",
            maxRAM=max_ram,
            n_cpu=n_cpu,
            )

    elif input_dict["inversion_type"].lower() == 'mag':
        prob = PF.Magnetics.MagneticIntegral(
            meshLocal, chiMap=tileMap, actInd=activeCells_t,
            parallelized=parallelized,
            Jpath=outDir + "Tile" + str(ind) + ".zarr",
            maxRAM=max_ram,
            n_cpu=n_cpu,
            )

    elif input_dict["inversion_type"].lower() in ['mvi', 'mvis']:
        prob = PF.Magnetics.MagneticIntegral(
            meshLocal, chiMap=tileMap, actInd=activeCells_t,
            parallelized=parallelized,
            Jpath=outDir + "Tile" + str(ind) + ".zarr",
            maxRAM=max_ram,
            modelType='vector',
            n_cpu=n_cpu
        )

    survey_t.pair(prob)

    # Data misfit function
    dmis = DataMisfit.l2_DataMisfit(survey_t)
    dmis.W = 1./survey_t.std

    wr = prob.getJtJdiag(np.ones(tileMap.shape[1]), W=dmis.W)

    activeCellsTemp = Maps.InjectActiveCells(mesh, activeCells, 1e-8)

    Mesh.TreeMesh.writeUBC(
      mesh, outDir + 'Octree_Tile' + str(ind) + '.msh',
      models={outDir + 'JtJ_Tile' + str(ind) + ' .act': activeCellsTemp*wr[:nC]}
    )
    wrGlobal += wr

    del meshLocal

    # Create combo misfit function
    return dmis, wrGlobal


for ind, (local_mesh, local_survey) in enumerate(zip(local_meshes, local_surveys)):

    print("Tile " + str(tt+1) + " of " + str(X1.shape[0]))

    dmis, wrGlobal = createLocalProb(local_mesh, local_survey, wrGlobal, ind)

    # Add the problems to a Combo Objective function
    if ind == 0:
        ComboMisfit = dmis

    else:
        ComboMisfit += dmis

# Global sensitivity weights (linear)
wrGlobal = wrGlobal**0.5
wrGlobal = (wrGlobal/np.max(wrGlobal))


# Mesh.TreeMesh.writeUBC(
#       mesh, outDir + 'OctreeMeshGlobal.msh',
#       models={outDir + 'SensWeights.mod': activeCellsMap * wrGlobal}
#     )

if input_dict["inversion_type"].lower() in ['grav', 'mag']:
    # Create a regularization function
    reg = Regularization.Sparse(
        mesh, indActive=activeCells, mapping=idenMap,
        alpha_s=alphas[0],
        alpha_x=alphas[1],
        alpha_y=alphas[2],
        alpha_z=alphas[3]
        )
    reg.norms = np.c_[model_norms].T
    reg.mref = np.ones(nC) * model_reference[0]
    reg.cell_weights = wrGlobal
    mstart = np.ones(nC) * model_start[0]
else:
    if len(model_reference) == 3:
        mref = np.kron(model_reference, np.ones(nC))
    else:
        # Assumes amplitude reference, distributed on 3 components in inducing field direction
        mref = np.kron(model_reference * Utils.matutils.dipazm_2_xyz(dip=survey.srcField.param[1], azm_N=survey.srcField.param[2])[0,:], np.ones(nC))

    if len(model_start) == 3:
        mstart = np.kron(model_start, np.ones(nC))
    else:
        # Assumes amplitude reference, distributed on 3 components in inducing field direction
        mstart = np.kron(model_start * Utils.matutils.dipazm_2_xyz(dip=survey.srcField.param[1], azm_N=survey.srcField.param[2])[0,:], np.ones(nC))

    # Create a block diagonal regularization
    wires = Maps.Wires(('p', nC), ('s', nC), ('t', nC))

    # Create a regularization
    reg_p = Regularization.Sparse(mesh, indActive=activeCells, mapping=wires.p)
    reg_p.alphas = alphas
    reg_p.cell_weights = (wires.p * wrGlobal)
    reg_p.norms = np.c_[2, 2, 2, 2]
    reg_p.mref = mref

    reg_s = Regularization.Sparse(mesh, indActive=activeCells, mapping=wires.s)
    reg_s.alphas = alphas
    reg_s.cell_weights = (wires.s * wrGlobal)
    reg_s.norms = np.c_[2, 2, 2, 2]
    reg_s.mref = mref

    reg_t = Regularization.Sparse(mesh, indActive=activeCells, mapping=wires.t)
    reg_t.alphas = alphas
    reg_t.cell_weights = (wires.t * wrGlobal)
    reg_t.norms = np.c_[2, 2, 2, 2]
    reg_t.mref = mref

    # Assemble the 3-component regularizations
    reg = reg_p + reg_s + reg_t
    reg.mref = mref

# Specify how the optimization will proceed, set susceptibility bounds to inf
opt = Optimization.ProjectedGNCG(maxIter=25, lower=-np.inf,
                                 upper=np.inf, maxIterLS=20,
                                 maxIterCG=30, tolCG=1e-3)

# Create the default L2 inverse problem from the above objects
invProb = InvProblem.BaseInvProblem(ComboMisfit, reg, opt)

# Specify how the initial beta is found
# if input_dict["inversion_type"].lower() in ['mvi', 'mvis']:
betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e+1)

# Pre-conditioner
update_Jacobi = Directives.UpdatePreconditioner()

IRLS = Directives.Update_IRLS(
                        f_min_change=1e-3, minGNiter=1, beta_tol=0.25,
                        maxIRLSiter=max_IRLS_iter, chifact_target=target_chi,
                        betaSearch=False)

# Save model
saveDict = Directives.SaveOutputEveryIteration(save_txt=False)
saveIt = Directives.SaveUBCModelEveryIteration(
    mapping=activeCellsMap, fileName=outDir + input_dict["inversion_type"].lower() + "_C",
    vector=input_dict["inversion_type"].lower()[0:3] == 'mvi'
)

# Put all the parts together
inv = Inversion.BaseInversion(invProb,
                              directiveList=[saveIt, saveDict, betaest, IRLS, update_Jacobi])

# SimPEG reports half phi_d, so we scale to matrch
print("Start Inversion\nTarget Misfit: %.2e (%.0f data with chifact = %g)" % (0.5*target_chi*len(survey.std), len(survey.std), target_chi))

# Run the inversion
mrec = inv.run(mstart)

print("Target Misfit: %.3e (%.0f data with chifact = %g)" % (0.5*target_chi*len(survey.std), len(survey.std), target_chi))
print("Final Misfit:  %.3e" % (0.5 * np.sum(((survey.dobs - invProb.dpred)/survey.std)**2.)))

if show_graphics:
    # Plot convergence curves
    fig, axs = plt.figure(), plt.subplot()
    axs.plot(saveDict.phi_d, 'ko-', lw=2)
    phi_d_target = 0.5*target_chi*len(survey.std)
    left, right = plt.xlim()
    axs.plot(
        np.r_[left, right],
        np.r_[phi_d_target, phi_d_target], 'r--'
    )
    plt.yscale('log')
    
    twin = axs.twinx()
    twin.plot(saveDict.phi_m, 'k--', lw=2)
    plt.autoscale(enable=True, axis='both', tight=True)        

    axs.set_ylabel('$\phi_d$', size=16, rotation=0)
    axs.set_xlabel('Iterations', size=14)
    twin.set_ylabel('$\phi_m$', size=16, rotation=0)
    t = fig.get_size_inches()
    fig.set_size_inches(t[0]*2, t[1]*2)
    fig.savefig(outDir + 'Convergence_curve.png', bbox_inches='tight', dpi=600)
    plt.show(block=False)

if getattr(ComboMisfit, 'objfcts', None) is not None:
    dpred = np.zeros(survey.nD)
    for ind, dmis in enumerate(ComboMisfit.objfcts):
        dpred[dmis.survey.ind] += dmis.survey.dpred(mrec).compute()
else:
    dpred = ComboMisfit.survey.dpred(mrec)

if input_dict["inversion_type"].lower() == 'grav':

    Utils.io_utils.writeUBCgravityObservations(outDir + 'Predicted.pre', survey, dpred+d0)

elif input_dict["inversion_type"].lower() in ['mvi', 'mvis', 'mag']:

    Utils.io_utils.writeUBCmagneticsObservations(outDir + 'Predicted.pre', survey, dpred+d0)

# Repeat inversion in spherical
if input_dict["inversion_type"].lower() == 'mvis':
    # Extract the vector components for the MVI-S
    x = activeCellsMap * (wires.p * invProb.model)
    y = activeCellsMap * (wires.s * invProb.model)
    z = activeCellsMap * (wires.t * invProb.model)

    amp = (np.sum(np.c_[x, y, z]**2., axis=1))**0.5

    # Get predicted data for each tile and form/write full predicted to file
    if getattr(ComboMisfit, 'objfcts', None) is not None:
        dpred = np.zeros(survey.nD)
        for ind, dmis in enumerate(ComboMisfit.objfcts):
            dpred[dmis.survey.ind] += np.asarray(dmis.survey.dpred(mrec))
    else:
        dpred = ComboMisfit.survey.dpred(mrec)

    Utils.io_utils.writeUBCmagneticsObservations(
      outDir + 'MVI_C_pred.pre', survey, dpred+d0
    )

    beta = invProb.beta

    # Change the starting model from Cartesian to Spherical
    mstart = Utils.matutils.xyz2atp(mrec.reshape((nC, 3), order='F'))
    mref = Utils.matutils.xyz2atp(mref.reshape((nC, 3), order='F'))

    # Flip the problem from Cartesian to Spherical
    if getattr(ComboMisfit, 'objfcts', None) is not None:
        for misfit in ComboMisfit.objfcts:
            misfit.prob.coordinate_system = 'spherical'
            misfit.prob.model = mstart
    else:
        ComboMisfit.prob.coordinate_system = 'spherical'
        ComboMisfit.prob.model = mstart

    # Create a block diagonal regularization
    wires = Maps.Wires(('amp', nC), ('theta', nC), ('phi', nC))

    # Create a regularization
    reg_a = Regularization.Sparse(mesh, indActive=activeCells,
                                  mapping=wires.amp, gradientType=gradient_type)
    reg_a.norms = model_norms[:4].T
    reg_a.mref = mref

    reg_t = Regularization.Sparse(mesh, indActive=activeCells,
                                  mapping=wires.theta, gradientType=gradient_type)
    reg_t.alpha_s = 0   # No reference angle
    reg_t.space = 'spherical'
    reg_t.norms = model_norms[4:8].T
    reg_t.mref = mref

    reg_p = Regularization.Sparse(mesh, indActive=activeCells,
                                  mapping=wires.phi, gradientType=gradient_type)
    reg_p.alpha_s = 0  # No reference angle
    reg_p.space = 'spherical'
    reg_p.norms = model_norms[8:].T
    reg_p.mref = mref

    # Assemble the three regularization
    reg = reg_a + reg_t + reg_p
    reg.mref = mref

    Lbound = np.kron(np.asarray([0, -np.inf, -np.inf]), np.ones(nC))
    Ubound = np.kron(np.asarray([10, np.inf, np.inf]), np.ones(nC))


    # Add directives to the inversion
    opt = Optimization.ProjectedGNCG(maxIter=40,
                                     lower=Lbound,
                                     upper=Ubound,
                                     maxIterLS=20,
                                     maxIterCG=30, tolCG=1e-3,
                                     stepOffBoundsFact=1e-8,
                                     LSshorten=0.25)

    invProb = InvProblem.BaseInvProblem(ComboMisfit, reg, opt, beta=beta*1)
    #  betaest = Directives.BetaEstimate_ByEig()

    # Here is where the norms are applied
    IRLS = Directives.Update_IRLS(f_min_change=1e-4, maxIRLSiter=max_IRLS_iter,
                                  minGNiter=1, beta_tol=0.3, prctile=100,
                                  coolingRate=1, coolEps_q=True,
                                  betaSearch=False)

    # Special directive specific to the mag amplitude problem. The sensitivity
    # weights are update between each iteration.
    ProjSpherical = Directives.ProjSpherical()
    update_SensWeight = Directives.UpdateSensitivityWeights()
    update_Jacobi = Directives.UpdatePreconditioner()
    saveDict = Directives.SaveOutputEveryIteration(save_txt=False)
    saveModel = Directives.SaveUBCModelEveryIteration(mapping=activeCellsMap, vector=True)
    saveModel.fileName = outDir + input_dict["inversion_type"].lower() + "_S"

    inv = Inversion.BaseInversion(invProb,
                                  directiveList=[
                                    ProjSpherical, IRLS, update_SensWeight,
                                    update_Jacobi, saveModel, saveDict
                                    ])

    # Run the inversion
    print("Run Spherical inversion")
    mrec_S = inv.run(mstart)

    print("Target Misfit: %.3e (%.0f data with chifact = %g)" % (0.5*target_chi*len(survey.std), len(survey.std), target_chi))
    print("Final Misfit:  %.3e" % (0.5 * np.sum(((survey.dobs - invProb.dpred)/survey.std)**2.)))

    if show_graphics:
        # Plot convergence curves
        fig, axs = plt.figure(), plt.subplot()
        axs.plot(saveDict.phi_d, 'ko-', lw=2)
        phi_d_target = 0.5*target_chi*len(survey.std)
        left, right = plt.xlim()
        axs.plot(
            np.r_[left, right],
            np.r_[phi_d_target, phi_d_target], 'r--'
        )
    
        plt.yscale('log')
        bottom, top = plt.ylim()
        axs.plot(
            np.r_[IRLS.iterStart, IRLS.iterStart],
            np.r_[bottom, top], 'k:'
        )
        
        twin = axs.twinx()
        twin.plot(saveDict.phi_m, 'k--', lw=2)
        plt.autoscale(enable=True, axis='both', tight=True)        
        axs.text(
            IRLS.iterStart, top,
            'IRLS', va='top', ha='center',
            rotation='vertical', size=12,
            bbox={'facecolor': 'white'}
        )
    
        axs.set_ylabel('$\phi_d$', size=16, rotation=0)
        axs.set_xlabel('Iterations', size=14)
        twin.set_ylabel('$\phi_m$', size=16, rotation=0)
        t = fig.get_size_inches()
        fig.set_size_inches(t[0]*2, t[1]*2)
        fig.savefig(outDir + 'Convergence_curve_spherical.png', bbox_inches='tight', dpi=600)
        plt.show(block=False)
    
    if getattr(ComboMisfit, 'objfcts', None) is not None:
        dpred = np.zeros(survey.nD)
        for ind, dmis in enumerate(ComboMisfit.objfcts):
            dpred[dmis.survey.ind] += dmis.survey.dpred(mrec_S).compute()
    else:
        dpred = ComboMisfit.survey.dpred(mrec_S)
    
    Utils.io_utils.writeUBCmagneticsObservations(outDir + 'Predicted_MVI_S.pre', survey, dpred+d0)

# Ouput result
# Mesh.TreeMesh.writeUBC(
#       mesh, outDir + 'OctreeMeshGlobal.msh',
#       models={outDir + input_dict["inversion_type"].lower() + '.mod': activeCellsMap * invProb.model}
#     )


###############################################################################
# FORWARD

if "forward" in list(input_dict.keys()):
    if input_dict["forward"][0].lower() == "drape":
        print("DRAPED")
        # Define an octree mesh based on the data
        nx = int((rxLoc[:, 0].max()-rxLoc[:, 0].min()) / input_dict["forward"][1])
        ny = int((rxLoc[:, 1].max()-rxLoc[:, 1].min()) / input_dict["forward"][2])
        vectorX = np.linspace(rxLoc[:, 0].min(), rxLoc[:, 0].max(), nx)
        vectorY = np.linspace(rxLoc[:, 1].min(), rxLoc[:, 1].max(), ny)

        x, y = np.meshgrid(vectorX, vectorY)

        # Only keep points within max distance
        tree = cKDTree(np.c_[rxLoc[:, 0], rxLoc[:, 1]])
        # xi = _ndim_coords_from_arrays(, ndim=2)
        dists, indexes = tree.query(np.c_[mkvc(x), mkvc(y)])

        x = mkvc(x)[dists < input_dict["forward"][4]]
        y = mkvc(y)[dists < input_dict["forward"][4]]

        z = F(mkvc(x), mkvc(y)) + input_dict["forward"][3]
        newLocs = np.c_[mkvc(x), mkvc(y), mkvc(z)]

    elif input_dict["forward"][0].lower() == "upwardcontinuation":
        newLocs = rxLoc.copy()
        newLocs[:, 2] += input_dict["forward"][1]

    if input_dict["inversion_type"].lower() == 'grav':
        rxLoc = PF.BaseGrav.RxObs(newLocs)
        srcField = PF.BaseGrav.SrcField([rxLoc])
        forward = PF.BaseGrav.LinearSurvey(srcField)

    elif input_dict["inversion_type"].lower() in ['mvi', 'mag']:
        rxLoc = PF.BaseMag.RxObs(newLocs)
        srcField = PF.BaseMag.SrcField([rxLoc], param=survey.srcField.param)
        forward = PF.BaseMag.LinearSurvey(srcField)

    forward.std = np.ones(newLocs.shape[0])

    activeGlobal = (activeCellsMap * invProb.model) != ndv
    idenMap = Maps.IdentityMap(nP=int(activeGlobal.sum()))

    if input_dict["inversion_type"].lower() == 'grav':
        fwrProb = PF.Gravity.GravityIntegral(
            mesh, rhoMap=idenMap, actInd=activeCells,
            n_cpu=n_cpu, forwardOnly=True, rxType='xyz'
            )
    elif input_dict["inversion_type"].lower() in ['mvi', 'mag']:
        fwrProb = PF.Magnetics.MagneticIntegral(
            mesh, chiMap=idenMap, actInd=activeCells,
            n_cpu=n_cpu, forwardOnly=True, rxType='xyz'
            )

    forward.pair(fwrProb)
    pred = fwrProb.fields(invProb.model)

    if input_dict["inversion_type"].lower() == 'grav':

        Utils.io_utils.writeUBCgravityObservations(outDir + 'Forward.dat', forward, pred)

    elif input_dict["inversion_type"].lower() in ['mvi', 'mvis', 'mag']:

        Utils.io_utils.writeUBCmagneticsObservations(outDir + 'Forward.dat', forward, pred)


if "eqs_mvi" in list(input_dict.keys()) and input_dict["inversion_type"].lower() in ['mvi', 'mvis']:
    # MAG ONLY RTP Amplitude
    print("Run RTP forward model")
    # Add a constant height to the existing locations for upward continuation
    newLocs = survey.srcField.rxList[0].locs
    newLocs[:, 2] += upward_continue;

    # Mag only
    rxLocNew = PF.BaseMag.RxObs(newLocs)
    # retain TF, but update inc-dec to vertical field
    srcField = PF.BaseMag.SrcField([rxLocNew], param=[survey.srcField.param[0],90,0])
    forward = PF.BaseMag.LinearSurvey(srcField, components=['tmi'])

    # Set unity standard deviations (required but not used)
    forward.std = np.ones(newLocs.shape[0])

    if input_dict["inversion_type"].lower() in ['mvi']:
        # Model lp out MVI_C
        vec_xyz = mrec.reshape((nC, 3), order='F')
    else:
       # Model lp out MVI_S
        vec_xyz = Utils.matutils.atp2xyz(
            mrec_S.reshape((nC, 3), order='F')).reshape((nC, 3), order='F')

    # RTP forward
    amp = (vec_xyz[:, 0]**2. + vec_xyz[:, 1]**2 + vec_xyz[:, 2]**2)**0.5
    activeGlobal = (activeCellsMap * amp) != ndv
    idenMap = Maps.IdentityMap(nP=int(activeGlobal.sum()))
    start_time = time.time()
    fwrProb = PF.Magnetics.MagneticIntegral(
        mesh, chiMap=idenMap, actInd=activeGlobal, parallelized='dask',
        forwardOnly=True
        )

    forward.pair(fwrProb)
    pred = fwrProb.fields(amp)
    elapsed_time = time.time() - start_time
    print('Time: {0:.3f} sec'.format(elapsed_time))
    
    Utils.io_utils.writeUBCmagneticsObservations(outDir + 'EQS_AMP_RTP.mag', forward, pred)
    if geosoft_output:
        t = GsftGrid_values[:,:,3]
        t[GsftGrid_mask[:,:,3]] = pred
        
        # Create output filenames        
        filenameParts = gxgrid.name_parts(input_dict["data_file"])
        outputGridName = os.path.join(outDir, 'EQS_AMP_RTP.grd(GRD;COMP=SPEED)')      
        newGrid = gxgrid.Grid.from_data_array(t, file_name=outputGridName, overwrite=True, properties=GsftGrid_props)
        newGrid.close()
        
    forward.unpair()
    forward = PF.BaseMag.LinearSurvey(srcField, components=['bx','by','bz'])
        # Set unity standard deviations (required but not used)
    forward.std = np.ones(newLocs.shape[0])

    # MAG ONLY RTP Amplitude
    print("Run Vector forward model")
    
    idenMap = Maps.IdentityMap(nP=3*int(activeGlobal.sum()))
    start_time = time.time()

    if input_dict["inversion_type"].lower() in ['mvi']:
        fwrProb = PF.Magnetics.MagneticIntegral(
            mesh, chiMap=idenMap, actInd=activeGlobal, parallelized='dask',
            forwardOnly=True, modelType='vector'
            )
    
        forward.pair(fwrProb)
        pred = fwrProb.fields(mrec)
    else:
        fwrProb = PF.Magnetics.MagneticIntegral(
            mesh, chiMap=idenMap, actInd=activeGlobal, parallelized='dask',
            forwardOnly=True, coordinate_system = 'spherical', modelType='vector'
            )
    
        forward.pair(fwrProb)
        pred = fwrProb.fields(mrec_S)
        
    elapsed_time = time.time() - start_time
    print('Time: {0:.3f} sec'.format(elapsed_time))

    d_amp = np.sqrt(pred[::3]**2. +
                    pred[1::3]**2. +
                    pred[2::3]**2.)

    Utils.io_utils.writeUBCmagneticsObservations(outDir + 'EQS_MVI_Bx.mag', forward, pred[::3])
    Utils.io_utils.writeUBCmagneticsObservations(outDir + 'EQS_MVI_By.mag', forward, pred[1::3])
    Utils.io_utils.writeUBCmagneticsObservations(outDir + 'EQS_MVI_Bz.mag', forward, pred[2::3])
    Utils.io_utils.writeUBCmagneticsObservations(outDir + 'EQS_MVI_RTP.mag', forward, -pred[2::3])
    Utils.io_utils.writeUBCmagneticsObservations(outDir + 'EQS_MVI_Amp.mag', forward, d_amp)

    if geosoft_output:
        t = GsftGrid_values[:,:,3]
        t[GsftGrid_mask[:,:,3]] = pred[::3]
        
        # Create output filenames        
        outputGridName = os.path.join(outDir, 'EQS_MVI_Bx.grd(GRD;COMP=SPEED)')      
        newGrid = gxgrid.Grid.from_data_array(t, file_name=outputGridName, overwrite=True, properties=GsftGrid_props)
        newGrid.close()

        t[GsftGrid_mask[:,:,3]] = pred[1::3]
        
        # Create output filenames        
        outputGridName = os.path.join(outDir, 'EQS_MVI_By.grd(GRD;COMP=SPEED)')      
        newGrid = gxgrid.Grid.from_data_array(t, file_name=outputGridName, overwrite=True, properties=GsftGrid_props)
        newGrid.close()

        t[GsftGrid_mask[:,:,3]] = pred[2::3]
        
        # Create output filenames        
        outputGridName = os.path.join(outDir, 'EQS_MVI_Bz.grd(GRD;COMP=SPEED)')      
        newGrid = gxgrid.Grid.from_data_array(t, file_name=outputGridName, overwrite=True, properties=GsftGrid_props)
        newGrid.close()

        t[GsftGrid_mask[:,:,3]] = -pred[2::3]
        
        # Create output filenames        
        outputGridName = os.path.join(outDir, 'EQS_MVI_RTP.grd(GRD;COMP=SPEED)')      
        newGrid = gxgrid.Grid.from_data_array(t, file_name=outputGridName, overwrite=True, properties=GsftGrid_props)
        newGrid.close()

        t[GsftGrid_mask[:,:,3]] = d_amp
        
        # Create output filenames        
        outputGridName = os.path.join(outDir, 'EQS_MVI_AMP.grd(GRD;COMP=SPEED)')      
        newGrid = gxgrid.Grid.from_data_array(t, file_name=outputGridName, overwrite=True, properties=GsftGrid_props)
        newGrid.close()
