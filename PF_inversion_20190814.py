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

# NEED TO ADD ALPHA VALUES
# NEED TO ADD REFERENCE
# NEED TO ADD STARTING

dsep = os.path.sep
input_file = sys.argv[1]

if input_file is not None:
    workDir = dsep.join(
                input_file.split(dsep)[:-1]
            )
    if len(workDir) > 0:
        workDir += dsep

else:

    assert input_file is not None, "The input file is missing: 'python PFinversion.py input_file.json'"

# Read json file and overwrite defaults
with open(input_file, 'r') as f:
    driver = json.load(f)

input_dict = dict((k.lower(), driver[k]) for k in list(driver.keys()))


if "result_folder" in list(input_dict.keys()):
    outDir = workDir + dsep + input_dict["result_folder"] + dsep
else:
    outDir = workDir + dsep + "SimPEG_PFInversion" + dsep


# Default parameter values
parallelized = 'dask'
meshType = 'TREE'
tileProblem = True

topo = False
ndv = -100

os.system('mkdir ' + outDir)
# Deal with the data
if input_dict["data_type"].lower() == 'ubc_grav':

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
else:

    assert False, "PF Inversion only implemented for 'data_type' 'ubc_grav' | 'ubc_mag' | 'ftmg' "

if survey.std is None:

    survey.std = survey.dobs * 0 + 10 #

if "mesh_file" in list(input_dict.keys()):
    meshInput = Mesh.TreeMesh.readUBC(workDir + input_dict["mesh_file"])
else:
    meshInput = None

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

if len(octree_levels_padding) < len(octree_levels_obs):
    octree_levels_padding += octree_levels_obs[len(octree_levels_padding):]

if "core_cell_size" in list(input_dict.keys()):
    core_cell_size = input_dict["core_cell_size"]
else:
    asser("'core_cell_size' must be added to the inputs")

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


if meshInput is None:
    print("Creating Global Octree")
    mesh = meshutils.mesh_builder_xyz(
        newTopo, core_cell_size,
        padding_distance=padding_distance,
        mesh_type='TREE', base_mesh=meshInput,
        depth_core=depth_core
        )

    if topo is not None:
        mesh = meshutils.refine_tree_xyz(
            mesh, topo, method='surface',
            octree_levels=octree_levels_topo, finalize=False
        )

    mesh = meshutils.refine_tree_xyz(
        mesh, rxLoc, method='surface',
        max_distance=max_distance,
        octree_levels=octree_levels_obs,
        octree_levels_padding=octree_levels_padding,
        finalize=True,
    )

else:
    mesh = meshInput

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
if input_dict["inversion_type"].lower() == 'mvi':
    wrGlobal = np.zeros(3*nC)
else:
    idenMap = Maps.IdentityMap(nP=nC)
    wrGlobal = np.zeros(nC)


# Loop over different tile size and break problem until
# memory footprint false below max_ram
usedRAM = np.inf
count = 1
while usedRAM > max_ram:
    print("Tiling:" + str(count))

    tiles, binCount, tileIDs, tile_numbers = Utils.modelutils.tileSurveyPoints(rxLoc, count)

    # Grab the smallest bin and generate a temporary mesh
    indMax = np.argmax(binCount)

    X1, Y1 = tiles[0][:, 0], tiles[0][:, 1]
    X2, Y2 = tiles[1][:, 0], tiles[1][:, 1]

    ind_t = tileIDs == tile_numbers[indMax]

    # Create the mesh and refine the same as the global mesh
    if count > 1:
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
    else:
        meshLocal = mesh

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

def createLocalProb(rxLoc, wrGlobal, ind_t, ind, singleMesh):
    # createLocalProb(rxLoc, wrGlobal, lims, ind)
    # Generate a problem, calculate/store sensitivities for
    # given data points

    # Grab the data for current tile
    # ind_t = np.all([rxLoc[:, 0] >= lims[0], rxLoc[:, 0] <= lims[1],
    #                 rxLoc[:, 1] >= lims[2], rxLoc[:, 1] <= lims[3],
    #                 surveyMask], axis=0)

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

    elif input_dict["inversion_type"].lower() in ['mag', 'mvi']:
        rxLoc_t = PF.BaseMag.RxObs(rxLoc[ind_t, :])
        srcField = PF.BaseMag.SrcField([rxLoc_t], param=survey.srcField.param)
        survey_t = PF.BaseMag.LinearSurvey(srcField, components=survey.components)

        dataInd = np.kron(ind_t, np.ones(len(survey.components))).astype('bool')

        survey_t.dobs = survey.dobs[dataInd]
        survey_t.std = survey.std[dataInd]
        survey_t.ind = ind_t

        # Utils.io_utils.writeUBCmagneticsObservations(outDir + "Tile" + str(ind) + '.dat', survey_t, survey_t.dobs)
    if singleMesh is False:
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
    else:

        print("Using global mesh")
        meshLocal = mesh


    # Need to find a way to compute sensitivities only for intersecting cells
    activeCells_t = np.ones(meshLocal.nC, dtype='bool')  # meshUtils.modelutils.activeTopoLayer(meshLocal, topo)

    # Create reduced identity map
    if input_dict["inversion_type"].lower() == 'mvi':
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

    elif input_dict["inversion_type"].lower() == 'mvi':
        prob = PF.Magnetics.MagneticIntegral(
            meshLocal, chiMap=tileMap, actInd=activeCells_t,
            parallelized=parallelized,
            Jpath=outDir + "Tile" + str(ind) + ".zarr",
            maxRAM=max_ram,
            modelType='vector',
            n_cpu=n_cpu
        )

    survey_t.pair(prob)

    # Write out local active and obs for validation
    Mesh.TreeMesh.writeUBC(
      meshLocal, outDir + 'Octree_Tile' + str(ind) + '.msh',
      models={outDir + 'ActiveGlobal_Tile' + str(ind) + ' .act': activeCells_t}
    )

    # Data misfit function
    dmis = DataMisfit.l2_DataMisfit(survey_t)
    dmis.W = 1./survey_t.std

    wr = prob.getJtJdiag(np.ones(tileMap.shape[1]), W=dmis.W)**0.5

    activeCellsTemp = Maps.InjectActiveCells(mesh, activeCells, 1e-8)

    Mesh.TreeMesh.writeUBC(
      mesh, outDir + 'Octree_Tile' + str(ind) + '.msh',
      models={outDir + 'JtJ_Tile' + str(ind) + ' .act': activeCellsTemp*wr[:nC]}
    )
    wrGlobal += wr

    del meshLocal

    # Create combo misfit function
    return dmis

# Loop through the tiles and generate all sensitivities
print("Number of tiles:" + str(nTiles))
for tt in range(nTiles):

    print("Tile " + str(tt+1) + " of " + str(X1.shape[0]))

    dmis = createLocalProb(rxLoc, wrGlobal, tileIDs==tile_numbers[tt], tt, nTiles==1)

    # Add the problems to a Combo Objective function
    if tt == 0:
        ComboMisfit = dmis

    else:
        ComboMisfit += dmis

actvGlobal = wrGlobal != 0
if actvGlobal.sum() < activeCells.sum():

    for ind, dmis in enumerate(ComboMisfit.objfcts):
        dmis.prob.chiMap.index = actvGlobal


# Global sensitivity weights (linear)
wrGlobal = wrGlobal
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
    reg.mref = np.zeros(nC)
    reg.cell_weights = wrGlobal
    mstart = np.zeros(nC)
else:
    mstart = np.ones(3*nC) * 1e-4

    # Assumes amplitude reference, distributed on 3 components
    mref = np.zeros(3*nC)

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
# if input_dict["inversion_type"].lower() == 'mvi':
betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e+1)

# Pre-conditioner
update_Jacobi = Directives.UpdatePreconditioner()

if (input_dict["inversion_type"].lower() == 'mvi') or (np.all(model_norms==2)):
    maxIRLSiter = 1

else:
    maxIRLSiter = 20

IRLS = Directives.Update_IRLS(
                        f_min_change=1e-3, minGNiter=1, beta_tol=0.25,
                        maxIRLSiter=maxIRLSiter, chifact_target=target_chi,
                        betaSearch=False)

# Save model
saveIt = Directives.SaveUBCModelEveryIteration(
    mapping=activeCellsMap, fileName=outDir + input_dict["inversion_type"].lower(),
    vector=input_dict["inversion_type"].lower() == 'mvi'
)

# Put all the parts together
inv = Inversion.BaseInversion(invProb,
                              directiveList=[saveIt, betaest, IRLS, update_Jacobi])

# Run the inversion
mrec = inv.run(mstart)

# Repeat inversion in spherical
if input_dict["inversion_type"].lower() == 'mvi':
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
      outDir + 'MVI_C_pred.pre', survey, dpred
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

    invProb = InvProblem.BaseInvProblem(ComboMisfit, reg, opt, beta=beta*3)
    #  betaest = Directives.BetaEstimate_ByEig()

    # Here is where the norms are applied
    IRLS = Directives.Update_IRLS(f_min_change=1e-4, maxIRLSiter=40,
                                  minGNiter=1, beta_tol=0.5, prctile=100,
                                  coolingRate=1, coolEps_q=True,
                                  betaSearch=False)

    # Special directive specific to the mag amplitude problem. The sensitivity
    # weights are update between each iteration.
    ProjSpherical = Directives.ProjSpherical()
    update_SensWeight = Directives.UpdateSensitivityWeights()
    update_Jacobi = Directives.UpdatePreconditioner()
    saveModel = Directives.SaveUBCModelEveryIteration(mapping=activeCellsMap, vector=True)
    saveModel.fileName = outDir + input_dict["inversion_type"].lower() + "_S"

    inv = Inversion.BaseInversion(invProb,
                                  directiveList=[
                                    ProjSpherical, IRLS, update_SensWeight,
                                    update_Jacobi, saveModel
                                    ])

    # Run the inversion
    mrec_S = inv.run(mstart)


# Ouput result
# Mesh.TreeMesh.writeUBC(
#       mesh, outDir + 'OctreeMeshGlobal.msh',
#       models={outDir + input_dict["inversion_type"].lower() + '.mod': activeCellsMap * invProb.model}
#     )

if getattr(ComboMisfit, 'objfcts', None) is not None:
    dpred = np.zeros(survey.nD)
    for ind, dmis in enumerate(ComboMisfit.objfcts):
        dpred[dmis.survey.ind] += dmis.survey.dpred(mrec).compute()
else:
    dpred = ComboMisfit.survey.dpred(mrec)

if input_dict["inversion_type"].lower() == 'grav':

    Utils.io_utils.writeUBCgravityObservations(outDir + 'Predicted.dat', survey, dpred)

elif input_dict["inversion_type"].lower() in ['mvi', 'mag']:

    Utils.io_utils.writeUBCmagneticsObservations(outDir + 'Predicted.dat', survey, dpred)


if "forward" in list(input_dict.keys()):
    if input_dict["forward"][0] == "DRAPE":
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

    elif input_dict["forward"][0] == "UpwardContinuation":
        newLocs = rxLoc.copy()
        newLocs[:, 2] += input_dict["forward"][1]

    if input_dict["inversion_type"].lower() == 'grav':
        rxLoc = PF.BaseGrav.RxObs(newLocs)
        srcField = PF.BaseGrav.SrcField([rxLoc])
        forward = PF.BaseGrav.LinearSurvey(srcField)

    elif input_dict["inversion_type"].lower() == 'mag':
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
    elif input_dict["inversion_type"].lower() == 'mag':
        fwrProb = PF.Magnetics.MagneticIntegral(
            mesh, chiMap=idenMap, actInd=activeCells,
            n_cpu=n_cpu, forwardOnly=True, rxType='xyz'
            )

    forward.pair(fwrProb)
    pred = fwrProb.fields(invProb.model)

    if input_dict["inversion_type"].lower() == 'grav':

        Utils.io_utils.writeUBCgravityObservations(outDir + 'Forward.dat', forward, pred)

    elif input_dict["inversion_type"].lower() == 'mag':

        Utils.io_utils.writeUBCmagneticsObservations(outDir + 'Forward.dat', forward, pred)
