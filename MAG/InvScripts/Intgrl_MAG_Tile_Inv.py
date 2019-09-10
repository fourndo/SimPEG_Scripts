"""

This script runs an Magnetic Amplitude Inversion (MAI) from TMI data.
Magnetic amplitude data are weakly sensitive to the orientation of
magnetization, and can therefore better recover the location and geometry of
magnetic bodies in the presence of remanence. The algorithm is inspired from
Li & Shearer (2008), with an added iterative sensitivity weighting strategy to
counter the vertical streatchin that the old code had

This is done in three parts:

1- TMI data are inverted for an equivalent source layer.

2-The equivalent source layer is used to predict component data -> amplitude

3- Amplitude data are inverted in 3-D for an effective susceptibility model

Created on December 7th, 2016

@author: fourndo@gmail.com

"""
from SimPEG import Mesh, Directives, Maps, InvProblem, Optimization
from SimPEG import DataMisfit, Inversion, Utils, Regularization
from SimPEG.Utils import mkvc
from discretize.utils import meshutils
import SimPEG.PF as PF
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree
from discretize.utils import closestPoints
import os

if __name__ == '__main__':

    #work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Kevitsa\\Modeling\\MAG\\"
    #work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\\Research\\Synthetic\\Block_Gaussian_topo\\"
    work_dir = "C:\\Users\\DominiqueFournier\\Dropbox\\Projects\\Synthetic\\Nut_Cracker\\"
    #work_dir = "C:\\Egnyte\\Private\\dominiquef\\Projects\\4559_CuMtn_ZTEM\\Modeling\\MAG\\A1_Fenton\\"
    # work_dir = 'C:\\Users\\DominiqueFournier\\Dropbox\\Projects\\Kevitsa\\Kevitsa\\Modeling\\MAG\\Airborne\\'
    # work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\CraigModel\\MAG\\"
    #work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Yukon\\Modeling\\MAG\\"
    #work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Synthetic\\Triple_Block_lined\\"
    #work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\TKC\\DIGHEM_TMI\\"

    out_dir = "SimPEG_Susc_TileInv\\"
    input_file = "SimPEG_MAG.inp"

    dsep = os.path.sep


    padLen = 1000
    maxRAM = 1.5
    depth_core = 200
    n_cpu = 4
    distMax = 200
    octreeObs = [3,5,10]
    octreeObs_XY = [3,3,0] #[5, 20, 5]  # Octree levels below observation points
    octreeTopo = [0,1] #[0, 0, 3]
    maxDist = 100
    meshType = 'TREE'


    # %%
    # Read in the input file which included all parameters at once
    # (mesh, topo, model, survey, inv param, etc.)
    driver = PF.MagneticsDriver.MagneticsDriver_Inv(work_dir + input_file)

    os.system('if not exist ' + work_dir + out_dir + ' mkdir ' + work_dir+out_dir)

    # Access the mesh and survey information
    meshInput = driver.mesh
    survey = driver.survey
    rxLoc = survey.rxLoc

    topo = None
    if driver.topofile is not None:
        topo = np.genfromtxt(driver.basePath + driver.topofile,
                             skip_header=1)
    else:
        # Grab the top coordinate and make a flat topo
        indTop = meshInput.gridCC[:, 2] == meshInput.vectorCCz[-1]
        topo = meshInput.gridCC[indTop, :]
        topo[:, 2] += meshInput.hz.min()/2. + 1e-8

    # # TILE THE PROBLEM
    # Define core mesh properties
    h = np.r_[meshInput.hx.min(), meshInput.hy.min(), meshInput.hz.min()]

    # LOOP THROUGH TILES
    surveyMask = np.ones(survey.nD, dtype='bool')
    # Going through all problems:
    # 1- Pair the survey and problem
    # 2- Add up sensitivity weights
    # 3- Add to the ComboMisfit

    # Create first mesh outside the parallel process
    padDist = np.r_[np.c_[padLen, padLen], np.c_[padLen, padLen], np.c_[padLen, 0]]

    if meshType != meshInput._meshType:
        print("Creating Global Octree")
        mesh = meshutils.mesh_builder_xyz(
            topo, h, padding_distance=padDist, mesh_type='TREE', base_mesh=meshInput,
            depth_core=depth_core
            )

        if topo is not None:
            mesh = meshutils.refine_tree_xyz(
                mesh, topo, method='surface',
                octree_levels=octreeTopo, finalize=False
            )

        mesh = meshutils.refine_tree_xyz(
            mesh, rxLoc, method='surface',
            max_distance=maxDist,
            octree_levels=octreeObs,
            octree_levels_padding=octreeObs_XY,
            finalize=True,
        )


        if topo is not None:
            actv = Utils.surface2ind_topo(mesh, topo)
        else:
            actv = np.zeros(mesh.nC, dtype='bool')
            print(meshInput.vectorNz[-1])
            actv[mesh.gridCC[:, 2] < meshInput.vectorNz[-1]] = True

        m0 = np.ones(actv.sum()) * 1e-4

        print("Writing global Octree to file" + work_dir + out_dir + 'OctreeMeshGlobal.msh')
        Mesh.TreeMesh.writeUBC(
              mesh, work_dir + out_dir + 'OctreeMeshGlobal.msh',
              models={work_dir + out_dir + 'ActiveGlobal.act': actv}
            )


    else:
        mesh = meshInput
        actv = np.zeros(mesh.nC, dtype='bool')
        actv[driver.activeCells] = True
        m0 = driver.m0  # Starting model


    wrGlobal = np.zeros(int(actv.sum()))
    actvMap = Maps.InjectActiveCells(mesh, actv, 1e-8)
    # Loop over different tile size and break problem until memory usage is preserved
    usedRAM = np.inf
    count = 3
    while usedRAM > maxRAM:
        print("Tiling:" + str(count))

        tiles, binCount, labels = Utils.modelutils.tileSurveyPoints(rxLoc, count)

        # Grab the smallest bin and generate a temporary mesh
        indMin = np.argmin(binCount)

        X1, Y1 = tiles[0][:, 0], tiles[0][:, 1]
        X2, Y2 = tiles[1][:, 0], tiles[1][:, 1]

        ind_t = np.all([rxLoc[:, 0] >= tiles[0][indMin, 0], rxLoc[:, 0] <= tiles[1][indMin, 0],
                        rxLoc[:, 1] >= tiles[0][indMin, 1], rxLoc[:, 1] <= tiles[1][indMin, 1],
                        surveyMask], axis=0)
        meshLocal = meshutils.mesh_builder_xyz(
            topo, h, padding_distance=padDist, mesh_type='TREE', base_mesh=meshInput,
            depth_core=depth_core
        )

        if topo is not None:
            meshLocal = meshutils.refine_tree_xyz(
                meshLocal, topo, method='surface',
                octree_levels=octreeTopo, finalize=False
            )

        meshLocal = meshutils.refine_tree_xyz(
            meshLocal, rxLoc[ind_t, :], method='surface',
            max_distance=maxDist,
            octree_levels=octreeObs,
            octree_levels_padding=octreeObs_XY,
            finalize=True,
        )

        nD, nC = ind_t.sum()*1., meshLocal.nC*1.

        nChunks = n_cpu # Number of chunks
        cSa, cSb = int(nD/nChunks), int(nC/nChunks) # Chunk sizes
        usedRAM = nD * nC * 8. * 1e-9
        count += 1
        print(nD, nC, usedRAM)

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

    nTiles = X1.shape[0]
    def createLocalProb(rxLoc, wrGlobal, lims, ind):

        # Grab the data for current tile
        ind_t = np.all([rxLoc[:, 0] >= lims[0], rxLoc[:, 0] <= lims[1],
                        rxLoc[:, 1] >= lims[2], rxLoc[:, 1] <= lims[3],
                        surveyMask], axis=0)

        # Remember selected data in case of tile overlap
        surveyMask[ind_t] = False

        # Create new survey
        rxLoc_t = PF.BaseMag.RxObs(rxLoc[ind_t, :])
        srcField = PF.BaseMag.SrcField([rxLoc_t], param=survey.srcField.param)
        survey_t = PF.BaseMag.LinearSurvey(srcField)
        survey_t.dobs = survey.dobs[ind_t]
        survey_t.std = survey.std[ind_t]
        survey_t.ind = ind_t

        meshLocal = meshutils.mesh_builder_xyz(
            topo, h, padding_distance=padDist, mesh_type='TREE', base_mesh=meshInput,
            depth_core=depth_core
        )

        if topo is not None:
            meshLocal = meshutils.refine_tree_xyz(
                meshLocal, topo, method='surface',
                octree_levels=octreeTopo, finalize=False
            )

        meshLocal = meshutils.refine_tree_xyz(
            meshLocal, rxLoc[ind_t, :], method='surface',
            max_distance=maxDist,
            octree_levels=octreeObs,
            octree_levels_padding=octreeObs_XY,
            finalize=True,
        )

        actv_t = np.ones(meshLocal.nC, dtype='bool')

        # Create reduced identity map
        tileMap = Maps.Tile((mesh, actv), (meshLocal, actv_t))

        actv_t = tileMap.activeLocal

        print(actv_t.sum(), meshLocal.nC)
        # Create the forward model operator
        prob = PF.Magnetics.MagneticIntegral(
            meshLocal, chiMap=tileMap, actInd=actv_t, parallelized='dask',
            Jpath=work_dir + out_dir + "Tile" + str(ind) + ".zarr",
            store_zarr=True)

        survey_t.pair(prob)

        # Data misfit function
        dmis = DataMisfit.l2_DataMisfit(survey_t)
        dmis.W = 1./survey_t.std

        wr = prob.getJtJdiag(np.ones(tileMap.P.shape[1]), W=dmis.W)
        localMap = Maps.InjectActiveCells(meshLocal, actv_t, 1e-8)
        Mesh.TreeMesh.writeUBC(
                          mesh, work_dir + out_dir + 'OctreeMesh' + str(tt) + '.msh',
                          models={work_dir + out_dir + 'Wr_' + str(tt) + '.act': actvMap * wr}
                        )

        Mesh.TreeMesh.writeUBC(
                          meshLocal, work_dir + out_dir + 'OctreeMesh' + str(tt) + '.msh',
                          models={work_dir + out_dir + 'Wr_' + str(tt) + '.act': localMap * tileMap * wr}
                        )
    #    localMap = Maps.InjectActiveCells(meshLocal, actv_t, 1e-8)
    #    Mesh.TreeMesh.writeUBC(
    #                      meshLocal, work_dir + out_dir + 'OctreeMesh' + str(tt) + '.msh',
    #                      models={work_dir + out_dir + 'Wr_' + str(tt) + '.act': localMap*wr}
    #                    )
    #
        wrGlobal += wr

        del meshLocal


        # Create combo misfit function
        return dmis, wrGlobal


    for tt in range(nTiles):

        print("Tile " + str(tt+1) + " of " + str(X1.shape[0]))

        dmis, wrGlobal = createLocalProb(rxLoc, wrGlobal, np.r_[X1[tt], X2[tt], Y1[tt], Y2[tt]], tt)

        if tt == 0:
            ComboMisfit = dmis

        else:
            ComboMisfit += dmis

    # print('Sum of all problems:' + str(probSize*1e-6) + ' Mb')
    # Scale global weights for regularization

    # Check if global mesh has regions untouched by local problem
    actvGlobal = wrGlobal != 0
    if actvGlobal.sum() < actv.sum():

        for ind, dmis in enumerate(ComboMisfit.objfcts):
            dmis.prob.chiMap.index = actvGlobal

    wrGlobal = wrGlobal[actvGlobal]**0.5
    wrGlobal = (wrGlobal/np.max(wrGlobal))



    #%% Create a regularization
    actvMap = Maps.InjectActiveCells(mesh, actv, 0)
    actv = np.all([actv, actvMap*actvGlobal], axis=0)
    actvMap = Maps.InjectActiveCells(mesh, actv, 1e-8)
    idenMap = Maps.IdentityMap(nP=int(np.sum(actv)))

    Mesh.TreeMesh.writeUBC(
              mesh, work_dir + out_dir + 'OctreeMeshGlobal.msh',
              models={work_dir + out_dir + 'SensWeights.mod': actvMap * wrGlobal}
            )


    reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
    reg.norms = np.c_[driver.lpnorms].T
    reg.cell_weights = wrGlobal
    reg.mref = np.ones(mesh.nC)[actv] * np.mean(driver.mref)

    # Add directives to the inversion
    opt = Optimization.ProjectedGNCG(maxIter=30, lower=0., upper=10.,
                                     maxIterLS=20, maxIterCG=20, tolCG=1e-4)
    invProb = InvProblem.BaseInvProblem(ComboMisfit, reg, opt)
    betaest = Directives.BetaEstimate_ByEig()

    # Here is where the norms are applied
    # Use pick a treshold parameter empirically based on the distribution of
    #  model parameters
    IRLS = Directives.Update_IRLS(f_min_change=1e-3, minGNiter=1,
                                  maxIRLSiter=10)

    IRLS.target = driver.survey.nD
    update_Jacobi = Directives.UpdatePreconditioner()
    saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap, fileName=work_dir + out_dir + "MAG_Tile")
    inv = Inversion.BaseInversion(invProb,
                                  directiveList=[betaest, saveModel,
                                                 IRLS, update_Jacobi])

    # Run the inversion
    mrec = inv.run(m0)


    #for ind, dmis in enumerate(ComboMisfit.objfcts):
    #
    #    Mesh.TreeMesh.writeUBC(dmis.prob.mesh, work_dir + out_dir +  "Tile" + str(ind) + ".msh",
    #                           models={work_dir + out_dir + "Tile" + str(ind) + ".sus": dmis.prob.chiMap * mrec})
        # Mesh.TensorMesh.writeUBC(prob.mesh,
        #                          work_dir + out_dir + "Tile" + str(ind) + ".msh")
        # Mesh.TensorMesh.writeModelUBC(prob.mesh,
        #                               work_dir + out_dir + "Tile" + str(ind) + ".sus", prob.mapping() * mrec)


    # Outputs
    if mesh._meshType == 'TENSOR':
        Mesh.TensorMesh.writeUBC(mesh, work_dir + out_dir + "MAG_Tile.msh")
        Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + "MAG_Tile_lp.sus",
                                      actvMap*invProb.model)
        Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + "MAG_Tile_l2.sus",
                                      actvMap*invProb.l2model)
    else:
        Mesh.TreeMesh.writeUBC(mesh, work_dir + out_dir + 'MAG_Tile.msh',
                               models={work_dir + out_dir + 'Model_l2.sus': actvMap*invProb.l2model})
        Mesh.TreeMesh.writeUBC(mesh, work_dir + out_dir + 'MAG_Tile.msh',
                               models={work_dir + out_dir + 'Model_lp.sus': actvMap*mrec})


    # Get predicted data for each tile and write full predicted to file
    if getattr(ComboMisfit, 'objfcts', None) is not None:
        dpred = np.zeros(survey.nD)
        for ind, dmis in enumerate(ComboMisfit.objfcts):
            dpred[dmis.survey.ind] += dmis.survey.dpred(mrec)
    else:
        dpred = ComboMisfit.survey.dpred(mrec)
        # PF.Magnetics.writeUBCobs(work_dir+out_dir + "Tile" + str(ind) + ".pre",
        #                          survey, survey.dpred(mrec))

    Utils.io_utils.writeUBCmagneticsObservations(
        work_dir+out_dir + "Predicted_lp.pre", survey, dpred
        )
    Utils.io_utils.writeUBCmagneticsObservations(
        work_dir + out_dir + 'Predicted_l2.pre', survey,
        survey.dpred(invProb.l2model)
        )

    # # %% OUTPUT models for each tile
    # model = Mesh.TensorMesh.readModelUBC(driver.mesh,
    #                                      work_dir + '\Effec_sus_20mGrid.sus')
