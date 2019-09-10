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
from scipy.spatial import Delaunay
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import cKDTree
from SimPEG.Utils import mkvc
import dask
from dask.distributed import Client
import multiprocessing


#    client = Client(n_workers=4, threads_per_worker=1)
#    client
# dask.config.set(scheduler='threads')
# pool = multiprocessing.pool.ThreadPool(8)
# dask.config.set(pool=pool)

workDir = ".\\Assets\\MAG"
outDir = "SimPEG_PFInversion\\"
inputFile = "SimPEG_PFInversion.json"

# Default parameter values
padLen = 1000
maxRAM = 1.
n_cpu = 8
n_chunks = 1
parallelized = 'dask'
octreeObs = [5, 5]  # Octree levels below observation points
octreeLevels_XY = [5, 5]  # Number of octree cells padding
octreeTopo = [0, 0, 2] # Octree cells below topo


h = np.r_[20, 20, 20] # Default minimum cell size
meshType = 'TREE'
targetChi = 100
tileProblem = True
meshInput = None
padRatio = 0.
topo = False
ndv = -100

dsep = os.path.sep

os.system('mkdir ' + workDir + dsep + dsep + outDir)

# Read json file and overwrite defaults
with open(workDir + dsep + inputFile, 'r') as f:
    driver = json.load(f)

# Deal with the data
if driver["dataFile"][0] == 'GRAV':

    survey = Utils.io_utils.readUBCgravityObservations(workDir + dsep + driver["dataFile"][1])

elif driver["dataFile"][0] == 'MAG':

    survey, H0 = Utils.io_utils.readUBCmagneticsObservations(workDir + dsep + driver["dataFile"][1])

else:
    assert False, "PF Inversion only implemented for 'dataFile' 'GRAV' | 'MAG' "

if "mesh" in list(driver.keys()):
    meshInput = Mesh.TensorMesh.readUBC(workDir + dsep + driver["mesh"])

if "topography" in list(driver.keys()):
    topo = np.genfromtxt(workDir + dsep + driver["topography"],
                         skip_header=1)
#        topo[:,2] += 1e-8
    # Compute distance of obs above topo
    F = NearestNDInterpolator(topo[:, :2], topo[:, 2])

else:
    # Grab the top coordinate and make a flat topo
    indTop = meshInput.gridCC[:, 2] == meshInput.vectorCCz[-1]
    topo = meshInput.gridCC[indTop, :]
    topo[:, 2] += meshInput.hz.min()/2. + 1e-8


if "targetChi" in list(driver.keys()):
    targetChi = driver["targetChi"]


if "octreeCellSize" in list(driver.keys()):
    h = driver["octreeCellSize"]

if "tileProblem" in list(driver.keys()):
    tileProblem = driver["tileProblem"]

rxLoc = survey.srcField.rxList[0].locs

# Create near obs topo
newTopo = np.c_[rxLoc[:, :2], F(rxLoc[:,:2])]

# LOOP THROUGH TILES
surveyMask = np.ones(survey.nD, dtype='bool')
# Going through all problems:
# 1- Pair the survey and problem
# 2- Add up sensitivity weights
# 3- Add to the ComboMisfit

# Create first mesh outside the parallel process
padDist = np.r_[np.c_[padLen, padLen], np.c_[padLen, padLen], np.c_[padLen, 0]]

print("Creating Global Octree")
mesh = Utils.modelutils.meshBuilder(
        newTopo, h, padDist, meshType='TREE', meshGlobal=meshInput,
        verticalAlignment='center'
    )

if topo is not None:
    mesh = Utils.modelutils.refineTree(
        mesh, topo, dtype='surface',
        octreeLevels=octreeTopo, finalize=False
    )

mesh = Utils.modelutils.refineTree(
    mesh, newTopo, dtype='surface',
    octreeLevels=octreeObs,
    octreeLevels_XY=octreeLevels_XY,
    finalize=True
)

# Compute active cells
activeCells = Utils.surface2ind_topo(mesh, topo)

#    activeCells = Utils.modelutils.activeTopoLayer(mesh, topo)

Mesh.TreeMesh.writeUBC(
      mesh, workDir + dsep + outDir + 'OctreeMeshGlobal.msh',
      models={workDir + dsep + outDir + 'ActiveSurface.act': activeCells}
    )


# Get the layer of cells directly below topo
#activeCells = Utils.actIndFull2layer(mesh, active)
nC = int(activeCells.sum())  # Number of active cells
print(nC)
# Create active map to go from reduce set to full
activeCellsMap = Maps.InjectActiveCells(mesh, activeCells, ndv)

# Create identity map
idenMap = Maps.IdentityMap(nP=nC)
wrGlobal = np.zeros(nC)

if tileProblem:

    # Loop over different tile size and break problem until
    # memory footprint false below maxRAM
    usedRAM = np.inf
    count = 1
    while usedRAM > maxRAM:
        print("Tiling:" + str(count))

        tiles, binCount = Utils.modelutils.tileSurveyPoints(rxLoc, count)

        # Grab the smallest bin and generate a temporary mesh
        indMin = np.argmin(binCount)

        X1, Y1 = tiles[0][:, 0], tiles[0][:, 1]
        X2, Y2 = tiles[1][:, 0], tiles[1][:, 1]

        ind_t = np.all([rxLoc[:, 0] >= tiles[0][indMin, 0], rxLoc[:, 0] <= tiles[1][indMin, 0],
                        rxLoc[:, 1] >= tiles[0][indMin, 1], rxLoc[:, 1] <= tiles[1][indMin, 1],
                        surveyMask], axis=0)

        # Create the mesh and refine the same as the global mesh
        meshLocal = Utils.modelutils.meshBuilder(
                newTopo, h, padDist, meshType='TREE', meshGlobal=meshInput,
                verticalAlignment='center'
            )

        if topo is not None:
            meshLocal = Utils.modelutils.refineTree(
                meshLocal, topo, dtype='surface',
                octreeLevels=octreeTopo, finalize=False
            )

        # Refine on local
        meshLocal = Utils.modelutils.refineTree(
            meshLocal, newTopo[ind_t,:], dtype='surface',
            octreeLevels=octreeObs,
            octreeLevels_XY=octreeLevels_XY,
            finalize=True
        )

#            tileLayer = Utils.modelutils.activeTopoLayer(meshLocal, topo)
        tileLayer = Utils.surface2ind_topo(meshLocal, topo)

        # Calculate approximate problem size
        nDt, nCt = ind_t.sum()*1., tileLayer.sum()*1.

        nChunks = n_cpu # Number of chunks
        cSa, cSb = int(nDt/nChunks), int(nCt/nChunks) # Chunk sizes
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

    def createLocalProb(rxLoc, wrGlobal, lims, ind):
        # createLocalProb(rxLoc, wrGlobal, lims, ind)
        # Generate a problem, calculate/store sensitivities for
        # given data points

        # Grab the data for current tile
        ind_t = np.all([rxLoc[:, 0] >= lims[0], rxLoc[:, 0] <= lims[1],
                        rxLoc[:, 1] >= lims[2], rxLoc[:, 1] <= lims[3],
                        surveyMask], axis=0)

        # Remember selected data in case of tile overlap
        surveyMask[ind_t] = False

        # Create new survey
        if driver["dataFile"][0] == 'GRAV':
            rxLoc_t = PF.BaseGrav.RxObs(rxLoc[ind_t, :])
            srcField = PF.BaseGrav.SrcField([rxLoc_t])
            survey_t = PF.BaseGrav.LinearSurvey(srcField)
            survey_t.dobs = survey.dobs[ind_t]
            survey_t.std = survey.std[ind_t]
            survey_t.ind = ind_t

            Utils.io_utils.writeUBCgravityObservations(workDir + dsep + outDir + "Tile" + str(ind) + '.dat', survey_t, survey_t.dobs)

        elif driver["dataFile"][0] == 'MAG':
            rxLoc_t = PF.BaseMag.RxObs(rxLoc[ind_t, :])
            srcField = PF.BaseMag.SrcField([rxLoc_t], param=survey.srcField.param)
            survey_t = PF.BaseMag.LinearSurvey(srcField)
            survey_t.dobs = survey.dobs[ind_t]
            survey_t.std = survey.std[ind_t]
            survey_t.ind = ind_t

            Utils.io_utils.writeUBCmagneticsObservations(workDir + dsep + outDir + "Tile" + str(ind) + '.dat', survey_t, survey_t.dobs)

        meshLocal = Utils.modelutils.meshBuilder(
            newTopo, h, padDist, meshType='TREE', meshGlobal=meshInput,
            verticalAlignment='center'
        )

        if topo is not None:
            meshLocal = Utils.modelutils.refineTree(
                meshLocal, topo, dtype='surface',
                octreeLevels=octreeTopo, finalize=False
            )

        # Refine the mesh around loc
        meshLocal = Utils.modelutils.refineTree(
            meshLocal, newTopo[ind_t, :], dtype='surface',
            octreeLevels=octreeObs,
            octreeLevels_XY=octreeLevels_XY,
            finalize=True
        )

        # Need to find a way to compute sensitivities only for intersecting cells
        activeCells_t = np.ones(meshLocal.nC, dtype='bool')  # meshUtils.modelutils.activeTopoLayer(meshLocal, topo)

        # Create reduced identity map
        tileMap = Maps.Tile((mesh, activeCells), (meshLocal, activeCells_t))

        activeCells_t = tileMap.activeLocal

        print(activeCells_t.sum(), meshLocal.nC)
        if driver["dataFile"][0] == 'GRAV':
            prob = PF.Gravity.GravityIntegral(
                meshLocal, rhoMap=tileMap, actInd=activeCells_t,
                parallelized=parallelized,
                Jpath=workDir + dsep + outDir + "Tile" + str(ind) + ".zarr",
                maxRAM=maxRAM/n_cpu,
                n_cpu=n_cpu,
                n_chunks=n_chunks
                )

        elif driver["dataFile"][0] == 'MAG':
            prob = PF.Magnetics.MagneticIntegral(
                meshLocal, chiMap=tileMap, actInd=activeCells_t,
                parallelized=parallelized,
                Jpath=workDir + dsep + outDir + "Tile" + str(ind) + ".zarr",
                maxRAM=maxRAM/n_cpu,
                n_cpu=n_cpu,
                n_chunks=n_chunks
                )

        survey_t.pair(prob)

        # Write out local active and obs for validation
        Mesh.TreeMesh.writeUBC(
          meshLocal, workDir + dsep + outDir + dsep + 'Octree_Tile' + str(ind) + '.msh',
          models={workDir + dsep + outDir + dsep + 'ActiveGlobal_Tile' + str(ind) + ' .act': activeCells_t}
        )

        if driver["dataFile"][0] == 'GRAV':

            Utils.io_utils.writeUBCgravityObservations(workDir + dsep + outDir + dsep + 'Tile' + str(ind) + '.dat', survey_t, survey_t.dobs)

        elif driver["dataFile"][0] == 'MAG':

            Utils.io_utils.writeUBCmagneticsObservations(workDir + dsep + outDir + dsep + 'Tile' + str(ind) + '.dat', survey_t, survey_t.dobs)

        # Data misfit function
        dmis = DataMisfit.l2_DataMisfit(survey_t)
        dmis.W = 1./survey_t.std

        wr = prob.getJtJdiag(np.ones(tileMap.P.shape[1]), W=dmis.W)

        wrGlobal += wr

        del meshLocal

        # Create combo misfit function
        return dmis

    # Loop through the tiles and generate all sensitivities
    for tt in range(nTiles):

        print("Tile " + str(tt+1) + " of " + str(X1.shape[0]))

        dmis = createLocalProb(rxLoc, wrGlobal, np.r_[X1[tt], X2[tt], Y1[tt], Y2[tt]], tt)

        # Add the problems to a Combo Objective function
        if tt == 0:
            ComboMisfit = dmis

        else:
            ComboMisfit += dmis

else:
    # Create static map
    if driver["dataFile"][0] == 'GRAV':
        prob = PF.Gravity.GravityIntegral(
            mesh, rhoMap=idenMap, actInd=activeCells, parallelized=parallelized,
            Jpath=workDir+outDir+"sensitivity.zarr",
            n_cpu=n_cpu,
            n_chunks=n_chunks
            )
    elif driver["dataFile"][0] == 'MAG':
        prob = PF.Magnetics.MagneticIntegral(
            mesh, chiMap=idenMap, actInd=activeCells, parallelized=parallelized,
            Jpath=workDir+outDir+"sensitivity.zarr",
            n_cpu=n_cpu,
            n_chunks=n_chunks
            )

    survey.pair(prob)

    # Data misfit function
    ComboMisfit = DataMisfit.l2_DataMisfit(survey)
    ComboMisfit.W = 1./survey.std

    wrGlobal += prob.getJtJdiag(np.ones(int(nC)), W=ComboMisfit.W)
    actvGlobal = activeCells


actvGlobal = wrGlobal != 0
if actvGlobal.sum() < activeCells.sum():

    for ind, dmis in enumerate(ComboMisfit.objfcts):
        dmis.prob.chiMap.index = actvGlobal

#    @dask.delayed
#    def rowSum(Combo):
#        sumIt = 0
#        for fct in Combo.objfcts:
#            wr = fct.prob.getJtJdiag(np.ones(fct.prob.Xn.shape[0]), W=fct.W)
#
#            sumIt += wr
#        return sumIt
#

# Global sensitivity weights (linear)
wrGlobal = wrGlobal**0.5
wrGlobal = (wrGlobal/np.max(wrGlobal))


Mesh.TreeMesh.writeUBC(
      mesh, workDir + dsep + outDir + 'OctreeMeshGlobal.msh',
      models={workDir + dsep + outDir + 'SensWeights.mod': activeCellsMap * wrGlobal}
    )

# Create a regularization function, in this case l2l2
reg = Regularization.Sparse(mesh, indActive=activeCells, mapping=idenMap)
reg.mref = np.zeros(nC)
reg.cell_weights = wrGlobal

# Specify how the optimization will proceed, set susceptibility bounds to inf
opt = Optimization.ProjectedGNCG(maxIter=25, lower=-np.inf,
                                 upper=np.inf, maxIterLS=20,
                                 maxIterCG=30, tolCG=1e-3)

# Create the default L2 inverse problem from the above objects
invProb = InvProblem.BaseInvProblem(ComboMisfit, reg, opt)

# Specify how the initial beta is found
betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1)

# Target misfit to stop the inversion,
# try to fit as much as possible of the signal, we don't want to lose anything
targetMisfit = Directives.TargetMisfit(chifact=targetChi)

# Pre-conditioner
update_Jacobi = Directives.UpdatePreconditioner()

IRLS = Directives.Update_IRLS(f_min_change=1e-3, minGNiter=1, beta_tol=0.25,
                          maxIRLSiter=1, chifact_target = targetChi)

# Save model
saveIt = Directives.SaveUBCModelEveryIteration(mapping=activeCellsMap, fileName=workDir + dsep + outDir + driver["dataFile"][0])
# Put all the parts together
inv = Inversion.BaseInversion(invProb,
                              directiveList=[saveIt, betaest, IRLS, update_Jacobi])

# Run the equivalent source inversion
mstart = np.zeros(nC)
mrec = inv.run(mstart)

# Ouput result
Mesh.TreeMesh.writeUBC(
      mesh, workDir + dsep + outDir + 'OctreeMeshGlobal.msh',
      models={workDir + dsep + outDir + driver["dataFile"][0] + '.mod': activeCellsMap * invProb.model}
    )

if getattr(ComboMisfit, 'objfcts', None) is not None:
    dpred = np.zeros(survey.nD)
    for ind, dmis in enumerate(ComboMisfit.objfcts):
        dpred[dmis.survey.ind] += dmis.survey.dpred(mrec).compute()
else:
    dpred = ComboMisfit.survey.dpred(mrec)

if driver["dataFile"][0] == 'GRAV':

    Utils.io_utils.writeUBCgravityObservations(workDir + dsep + outDir + 'Predicted.dat', survey, dpred)

elif driver["dataFile"][0] == 'MAG':

    Utils.io_utils.writeUBCmagneticsObservations(workDir + dsep + outDir + 'Predicted.dat', survey, dpred)


if "forward" in list(driver.keys()):
    if driver["forward"][0] == "DRAPE":
        print("DRAPED")
        # Define an octree mesh based on the data
        nx = int((rxLoc[:, 0].max()-rxLoc[:, 0].min()) / driver["forward"][1])
        ny = int((rxLoc[:, 1].max()-rxLoc[:, 1].min()) / driver["forward"][2])
        vectorX = np.linspace(rxLoc[:, 0].min(), rxLoc[:, 0].max(), nx)
        vectorY = np.linspace(rxLoc[:, 1].min(), rxLoc[:, 1].max(), ny)

        x, y = np.meshgrid(vectorX, vectorY)

        # Only keep points within max distance
        tree = cKDTree(np.c_[rxLoc[:, 0], rxLoc[:, 1]])
        # xi = _ndim_coords_from_arrays(, ndim=2)
        dists, indexes = tree.query(np.c_[mkvc(x), mkvc(y)])

        x = mkvc(x)[dists < driver["forward"][4]]
        y = mkvc(y)[dists < driver["forward"][4]]

        z = F(mkvc(x), mkvc(y)) + driver["forward"][3]
        newLocs = np.c_[mkvc(x), mkvc(y), mkvc(z)]

    elif driver["forward"][0] == "UpwardContinuation":
        newLocs = rxLoc.copy()
        newLocs[:, 2] += driver["forward"][1]

    if driver["dataFile"][0] == 'GRAV':
        rxLoc = PF.BaseGrav.RxObs(newLocs)
        srcField = PF.BaseGrav.SrcField([rxLoc])
        forward = PF.BaseGrav.LinearSurvey(srcField)

    elif driver["dataFile"][0] == 'MAG':
        rxLoc = PF.BaseMag.RxObs(newLocs)
        srcField = PF.BaseMag.SrcField([rxLoc], param=survey.srcField.param)
        forward = PF.BaseMag.LinearSurvey(srcField)

    forward.std = np.ones(newLocs.shape[0])

    activeGlobal = (activeCellsMap * invProb.model) != ndv
    idenMap = Maps.IdentityMap(nP=int(activeGlobal.sum()))
    if driver["dataFile"][0] == 'GRAV':
        fwrProb = PF.Gravity.GravityIntegral(
            mesh, rhoMap=idenMap, actInd=activeCells,
            n_cpu=n_cpu, forwardOnly=True, rxType='xyz'
            )
    elif driver["dataFile"][0] == 'MAG':
        fwrProb = PF.Magnetics.MagneticIntegral(
            mesh, chiMap=idenMap, actInd=activeCells,
            n_cpu=n_cpu, forwardOnly=True, rxType='xyz'
            )

    forward.pair(fwrProb)
    pred = fwrProb.fields(invProb.model)

    if driver["dataFile"][0] == 'GRAV':

        Utils.io_utils.writeUBCgravityObservations(workDir + dsep + outDir + 'Forward.dat', forward, pred)

    elif driver["dataFile"][0] == 'MAG':

        Utils.io_utils.writeUBCmagneticsObservations(workDir + dsep + outDir + 'Forward.dat', forward, pred)
