###############################################################################
# Tiled Gravity Inversion with Model Decomposition
# ------------------------------------------------
#
# This scripts inverts gravity data (gz) with mesh de-coupling
# strategy and model decomposition:
#
#  m = m_ho + m_he
#
# such that the solution is split into an homogeneous and an
# heterogeneous part. The input reference model is used to isolate
# geological (homogeneous) units. Just as conventional 3D inversion
# approach, the heterogeneous part can be regualized with sparse norms.
#
# Author: fourndo@gmail.com
# Created: Feb 2th, 2019
#

import os
import numpy as np
import pylab as plt
from scipy.spatial import cKDTree
from matplotlib.patches import Rectangle
from SimPEG import Mesh, Directives, Maps, InvProblem, Optimization, Utils
from SimPEG import DataMisfit, Inversion, Regularization
import SimPEG.PF as PF


if __name__ == '__main__':

    work_dir = ".\\Tester"
    inpfile = 'SimPEG_GRAV.inp'
    dsep = os.path.sep
    out_dir = work_dir + dsep + "SimPEG_GRAV_ModelDecomp" + dsep
    padLen = 100
    maxRAM = 0.05
    n_cpu = 4

    octreeObs = [10, 3, 3]  # Octree levels below observation points
    octreeTopo = [0, 1]
    ndv = -100
    meshType = 'TREE'
    tileProblem = True
    parallization = "dask"  # "dask" ||  "multiprocessing"

    os.system('if not exist ' + out_dir + ' mkdir ' + out_dir)

    # Choice for the homogeneous model
    useMrefValues = True

    # Read input file
    driver = PF.GravityDriver.GravityDriver_Inv(work_dir + dsep + inpfile)
    meshInput = driver.mesh
    survey = driver.survey
    actv = driver.activeCells
    topo = driver.topo
    rxLoc = survey.srcField.rxList[0].locs

    # Tile the forward problem
    tree = cKDTree(meshInput.gridCC)

    h = np.r_[meshInput.hx.min(), meshInput.hy.min(), meshInput.hz.min()]

    # Create a mesh outside the parallel process
    padDist = np.r_[
        np.c_[padLen, padLen],
        np.c_[padLen, padLen],
        np.c_[padLen, 0]
    ]
    
    # Check if the input mesh is the same class as what is requested on top
    if meshType != meshInput._meshType:
        print("Creating Global Octree")
        mesh = Utils.modelutils.meshBuilder(
                rxLoc, h, padDist, meshType='TREE', meshGlobal=meshInput,
                verticalAlignment='center'
            )

        if topo is not None:
            mesh = Utils.modelutils.refineTree(
                mesh, topo, dtype='surface',
                nCpad=octreeTopo, finalize=False
            )

        mesh = Utils.modelutils.refineTree(
            mesh, rxLoc, dtype='surface',
            nCpad=octreeObs, finalize=True
        )

        if topo is not None:
            actv = Utils.surface2ind_topo(mesh, topo)
        else:
            actv = np.zeros(mesh.nC, dtype='bool')
            print(meshInput.vectorNz[-1])
            actv[mesh.gridCC[:, 2] < meshInput.vectorNz[-1]] = True

        if isinstance(driver.mstart, float):
            m0 = np.ones(mesh.nC) * driver.mstart

        else:
            print("Interpolating the starting model")

            _, ind = tree.query(mesh.gridCC)

            m0 = driver.m0
            m0[m0 == ndv] = 0
            m0 = m0[ind]

        if isinstance(driver._mrefInput, float):
            mref = np.ones(mesh.nC) * driver._mrefInput

        else:
            print("Interpolating the reference model")
            _, ind = tree.query(mesh.gridCC)

            mref = driver.mref
            mref[mref == ndv] = 0
            mref = mref[ind]

        print(
            "Writing global Octree to file" +
            out_dir + 'OctreeMeshGlobal.msh'
        )
        Mesh.TreeMesh.writeUBC(
              mesh, out_dir + 'OctreeMeshGlobal.msh',
              models={out_dir + 'ActiveGlobal.act': actv}
            )

        # Create new inteprolation tree for the tiles
        tree = cKDTree(mesh.gridCC)

    else:
        mesh = meshInput
        actv = np.zeros(mesh.nC, dtype='bool')
        actv[driver.activeCells] = True
        actvMap = Maps.InjectActiveCells(mesh, actv, 0)

        m0 = driver.m0  # Starting model
        mref = driver.mref  # Starting model

    # Create active map to go from reduce set to full
    fullMap = Maps.InjectActiveCells(mesh, actv, 0)

    # Set the global homogeneous mapping
    m0 = m0[actv]
    mgeo = mref[actv]

    # Get unique geo units
    geoUnits = np.unique(mgeo).tolist()

    # Compute an a median value for each homogeneous units
    mUnit = np.asarray([np.median(mgeo[mgeo == unit]) for unit in geoUnits])

    # Build list of indecies for the geounits
    index = []
    nCunit = []
    for unit in geoUnits:
        index += [mgeo == unit]
        nCunit += [int((mgeo == unit).sum())]
    nC = len(index)

    # Creat reduced identity map
    homogMap = Maps.SurjectUnits(index)

    # Create a wire map to link the homogeneous and heterogeneous spaces
    wires = Maps.Wires(('homo', nC), ('hetero', int(actv.sum())))

    # Create Sum map for the global inverse
    sumMap = Maps.SumMap([homogMap*wires.homo, wires.hetero])

    # LOOP THROUGH TILES
    surveyMask = np.ones(survey.nD, dtype='bool')  # Keep track of locs used
    wrGlobal = np.zeros(int(nC + actv.sum()))  # Global sensitivity weights
    if tileProblem:

        # Loop over different tile size and break
        # problem to fit in memory
        usedRAM = np.inf
        count = 0
        while usedRAM > maxRAM:
            print("Tiling:" + str(count))

            tiles, binCount = Utils.modelutils.tileSurveyPoints(rxLoc, count)

            # Grab the smallest bin and generate a temporary mesh
            indMin = np.argmin(binCount)

            X1, Y1 = tiles[0][:, 0], tiles[0][:, 1]
            X2, Y2 = tiles[1][:, 0], tiles[1][:, 1]

            ind_t = np.all([
                rxLoc[:, 0] >= tiles[0][indMin, 0],
                rxLoc[:, 0] <= tiles[1][indMin, 0],
                rxLoc[:, 1] >= tiles[0][indMin, 1],
                rxLoc[:, 1] <= tiles[1][indMin, 1],
                surveyMask
                ], axis=0)

            # Create the mesh and refine the same as the global mesh
            meshLocal = Utils.modelutils.meshBuilder(
                rxLoc, h, padDist, meshType='TREE', meshGlobal=meshInput,
                verticalAlignment='center'
            )

            if topo is not None:
                meshLocal = Utils.modelutils.refineTree(
                    meshLocal, topo, dtype='surface',
                    nCpad=octreeTopo, finalize=False
                )

            meshLocal = Utils.modelutils.refineTree(
                meshLocal, rxLoc[ind_t, :], dtype='surface',
                nCpad=octreeObs, finalize=True
            )

            # Calculate approximate problem size
            nD_t, nC_t = ind_t.sum()*1., meshLocal.nC*1.

            nChunks = n_cpu  # Number of chunks
            usedRAM = nD_t * nC_t * 8. * 1e-9
            count += 1
            print("Number of data in tile: ", nD_t)
            print("Number of cells in tile: ", nC_t)
            print("Estimated memory: ",  usedRAM, " Gb")

        # Plot data and tiles
        fig, ax1 = plt.figure(), plt.subplot()
        Utils.PlotUtils.plot2Ddata(rxLoc, survey.dobs, ax=ax1)
        for ii in range(X1.shape[0]):
            ax1.add_patch(Rectangle((X1[ii], Y1[ii]),
                                    X2[ii]-X1[ii],
                                    Y2[ii]-Y1[ii],
                                    facecolor='none', edgecolor='k'))
        ax1.set_xlim([X1.min()-20, X2.max()+20])
        ax1.set_ylim([Y1.min()-20, Y2.max()+20])
        ax1.set_aspect('equal')
        plt.show()

        # Function to generate a mesh and forward problem from data extent
        def createLocalProb(rxLoc, wrGlobal, lims, tileID):

            # Grab the data for current tile
            ind_t = np.all([rxLoc[:, 0] >= lims[0], rxLoc[:, 0] <= lims[1],
                            rxLoc[:, 1] >= lims[2], rxLoc[:, 1] <= lims[3],
                            surveyMask], axis=0)

            # Remember selected data in case of tile overlap
            surveyMask[ind_t] = False

            # Create new survey
            rxLoc_t = PF.BaseGrav.RxObs(rxLoc[ind_t, :])
            srcField = PF.BaseGrav.SrcField([rxLoc_t])
            survey_t = PF.BaseGrav.LinearSurvey(srcField)
            survey_t.dobs = survey.dobs[ind_t]
            survey_t.std = survey.std[ind_t]
            survey_t.ind = ind_t

            meshLocal = Utils.modelutils.meshBuilder(
                rxLoc, h, padDist, meshType='TREE', meshGlobal=meshInput,
                verticalAlignment='center'
            )

            if topo is not None:
                meshLocal = Utils.modelutils.refineTree(
                    meshLocal, topo, dtype='surface',
                    nCpad=octreeTopo, finalize=False
                )

            # Refine the mesh around loc
            meshLocal = Utils.modelutils.refineTree(
                meshLocal, rxLoc[ind_t, :], dtype='surface',
                nCpad=octreeObs, finalize=True
            )

            actv_t = np.ones(meshLocal.nC, dtype='bool')

            Mesh.TreeMesh.writeUBC(
                  meshLocal,
                  out_dir + 'OctreeMesh' + str(tt) + '.msh',
                  models={
                    out_dir + 'Active' + str(tileID) + '.act':
                    actv_t
                    }
                )

            # Create reduced identity map
            tileMap = Maps.Tile((mesh, actv), (meshLocal, actv_t))

            actv_t = tileMap.activeLocal

            # Interpolate the geo model
            _, ind = tree.query(meshLocal.gridCC)

            mgeo_t = ((fullMap * mgeo)[ind])[actv_t]

            # Build list of indecies for the geounits
            index = []
            for unit in geoUnits:
                index += [mgeo_t == unit]

            # Creat reduced identity map
            homogMap_t = Maps.SurjectUnits(index)

            # Create Sum map
            sumMap = Maps.SumMap([homogMap_t*wires.homo, tileMap*wires.hetero])

            # Create the forward model operator
            prob = PF.Gravity.GravityIntegral(
                meshLocal, rhoMap=sumMap,
                actInd=actv_t,
                parallelized=parallization,
                Jpath=out_dir + "Tile" + str(tileID) + ".zarr",
                n_cpu=n_cpu
                )

            survey_t.pair(prob)

            # Data misfit function
            dmis = DataMisfit.l2_DataMisfit(survey_t)
            dmis.W = 1./survey_t.std

            wrGlobal += prob.getJtJdiag(np.ones(wires.homo.shape[1]), W=dmis.W)

            del meshLocal

            # Create combo misfit function
            return dmis, wrGlobal

        # Cycle through the tiled data and create forward problems
        for tt in range(X1.shape[0]):

            print("Tile " + str(tt+1) + " of " + str(X1.shape[0]))

            dmis, wrGlobal = createLocalProb(
                rxLoc, wrGlobal, np.r_[X1[tt], X2[tt], Y1[tt], Y2[tt]], tt
            )

            if tt == 0:
                ComboMisfit = dmis

            else:
                ComboMisfit += dmis
    else:

        prob = PF.Gravity.GravityIntegral(
            mesh, rhoMap=sumMap, actInd=actv, parallelized=parallization,
            Jpath=out_dir + "Sensitivity.zarr", n_cpu=n_cpu)

        survey.pair(prob)

        # Data misfit function
        ComboMisfit = DataMisfit.l2_DataMisfit(survey)
        ComboMisfit.W = 1./survey.std

        wrGlobal += prob.getJtJdiag(
                np.ones(wires.homo.shape[1]), W=ComboMisfit.W
        )
        actvGlobal = actv

    # Create augmented mstart and mref
    mref = np.r_[mUnit, mgeo*0]
    mstart = np.r_[mUnit, m0]

    # Normalize the sensitivity weights
    wrGlobal[wires.homo.index] /= (np.max((wires.homo*wrGlobal)))
    wrGlobal[wires.hetero.index] /= (np.max(wires.hetero*wrGlobal))
    wrGlobal = wrGlobal**0.5

    # # Create a regularization
    # For the homogeneous model
    regMesh = Mesh.TensorMesh([nC])

    reg_m1 = Regularization.Sparse(regMesh, mapping=wires.homo)
    # Scale by sensitivity and number of cells in each unit
    reg_m1.cell_weights = (wires.homo*wrGlobal) / np.r_[nCunit]  
    reg_m1.mref = mref

    # Regularization for the voxel model
    reg_m2 = Regularization.Sparse(mesh, indActive=actv, mapping=wires.hetero)
    reg_m2.cell_weights = wires.hetero*wrGlobal
    reg_m2.norms = np.c_[driver.lpnorms].T
    reg_m2.mref = mref

    reg = reg_m1 + reg_m2

    # Create the Gauss-Newton + GC solver and set some parameters
    opt = Optimization.ProjectedGNCG(maxIter=25, lower=driver.bounds[0],
                                     upper=driver.bounds[1],
                                     maxIterLS=20, maxIterCG=20,
                                     tolCG=1e-4)
    
    # Piece an objective function together
    invProb = InvProblem.BaseInvProblem(ComboMisfit, reg, opt)

    # Add directives for the initial beta, pre-conditioner, sparse norms ...
    betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1.)
    IRLS = Directives.Update_IRLS(
            f_min_change=1e-4, minGNiter=1, betaSearch=False
    )
    update_Jacobi = Directives.UpdatePreconditioner()
    saveDict = Directives.SaveOutputDictEveryIteration()
    
    # The inversion object runs it all together
    inv = Inversion.BaseInversion(invProb, directiveList=[
                                                    betaest, IRLS, saveDict,
                                                    update_Jacobi
                                                    ])
    # Run inversion
    mrec = inv.run(mstart)

    # Output the result with no-data-values
    outMap = Maps.InjectActiveCells(mesh, actv, ndv)

    if meshType == "TENSOR":
        m_lp = outMap*(sumMap*invProb.model)
        Mesh.TensorMesh.writeModelUBC(
            mesh, out_dir + 'Total_inv_lp.den', m_lp
        )

        m_lp = outMap*(homogMap*wires.homo*invProb.model)
        Mesh.TensorMesh.writeModelUBC(
            mesh, out_dir + 'Homoge_inv_lp.den', m_lp
        )

        m_lp = outMap*(wires.hetero*invProb.model)
        Mesh.TensorMesh.writeModelUBC(
            mesh, out_dir + 'Hetero_inv_lp.den', m_lp
        )

    else:

        # Write the total sum model
        m_lp = outMap*(sumMap*invProb.model)
        Mesh.TreeMesh.writeUBC(
            mesh,
            out_dir + "TreeMesh.msh",
            models={out_dir + 'Total_inv_lp.den': m_lp}
        )

        # Write the homogeneous model
        m_lp = outMap*(homogMap*wires.homo*invProb.model)
        Mesh.TreeMesh.writeUBC(
            mesh,
            out_dir + "TreeMesh.msh",
            models={out_dir + 'Homoge_inv_lp.den': m_lp}
        )

        # Write the heterogeneous model
        m_lp = outMap*(wires.hetero*invProb.model)
        Mesh.TreeMesh.writeUBC(
            mesh,
            out_dir + "TreeMesh.msh",
            models={out_dir + 'Hetero_inv_lp.den': m_lp}
        )

    # Write out the predicted
    if getattr(ComboMisfit, 'objfcts', None) is not None:
        dpred = np.zeros(survey.nD)
        for ind, dmis in enumerate(ComboMisfit.objfcts):
            dpred[dmis.survey.ind] += dmis.survey.dpred(mrec)
    else:
        dpred = ComboMisfit.survey.dpred(mrec)

    Utils.io_utils.writeUBCgravityObservations(
        out_dir + "Predicted_lp.pre", survey, dpred
        )
