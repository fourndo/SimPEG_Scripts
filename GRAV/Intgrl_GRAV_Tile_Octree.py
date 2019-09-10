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
import SimPEG.PF as PF
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree
from discretize.utils import closestPoints
import os
import multiprocessing

if __name__ == '__main__':

    work_dir = 'C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Kevitsa\\Modeling\\GRAV\\'
#    work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Synthetic\\Block_Gaussian_topo\\GRAV\\"

    inpfile = 'SimPEG_GRAV.inp'
    out_dir = "SimPEG_GRAV_Inv\\"
    dsep = '\\'
    meshType = 'TreeMesh'
    padLen = 3000
    dwnFact = .25
    maxNpoints = 2000
    numProcessors = 8

    # %%
    # Read in the input file which included all parameters at once
    # (mesh, topo, model, survey, inv param, etc.)
    driver = PF.GravityDriver.GravityDriver_Inv(work_dir + inpfile)

    os.system('if not exist ' + work_dir + out_dir + ' mkdir ' + work_dir+out_dir)

    # Access the mesh and survey information
    meshInput = driver.mesh
    survey = driver.survey
    xyzLocs = survey.srcField.rxList[0].locs.copy()

    # Add mag obs
#    surveyMag = PF.Magnetics.readMagneticsObservations("C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Kevitsa\\Modeling\\MAG\\Airborne\\VTEM_FLT40m_IGRF53550nT.dat")
#    xyzLocs = np.r_[xyzLocs, surveyMag.srcField.rxList[0].locs.copy()]
    
    topo = None
    if driver.topofile is not None:
        topo = np.genfromtxt(driver.basePath + driver.topofile,
                             skip_header=1)
    else:
        # Grab the top coordinate and make a flat topo
        indTop = meshInput.gridCC[:, 2] == meshInput.vectorCCz[-1]
        topo = meshInput.gridCC[indTop, :]
        topo[:, 2] += meshInput.hz.min()/2. + 1e-8

    # Define an octree mesh based on the provided tensor
    h = np.r_[meshInput.hx.min(), meshInput.hy.min(), meshInput.hz.min()]
    coreX, coreY, coreZ = meshInput.hx == h[0], meshInput.hy == h[1], meshInput.hz == h[2]
    padx, pady, padz = meshInput.hx[~coreX].sum(), meshInput.hy[~coreY].sum(), meshInput.hz[~coreZ].sum()

    padDist = np.r_[np.c_[padx, padx], np.c_[pady, pady], np.c_[padz, padz]]
    
    if meshType == 'TreeMesh':
        if isinstance(meshInput, Mesh.TensorMesh):


            print("Creating TreeMesh. Please standby...")
            mesh = Utils.modelutils.meshBuilder(topo, h, padDist,
                                                meshGlobal=meshInput,
                                                meshType='TREE',
                                                gridLoc='CC')

            mesh = Utils.modelutils.refineTree(mesh, topo, dtype='surface',
                                               nCpad=[0, 3, 2], finalize=False)

            mesh = Utils.modelutils.refineTree(mesh, xyzLocs, dtype='surface',
                                               nCpad=[10, 5, 5], finalize=True)

        else:
            mesh = Mesh.TreeMesh.readUBC(driver.basePath + driver.mshfile)
    else:
        mesh = meshInput
    actv = Utils.surface2ind_topo(mesh, topo)

    if isinstance(mesh, Mesh.TreeMesh):
        Mesh.TreeMesh.writeUBC(mesh, work_dir + out_dir + 'OctreeMesh.msh',
                               models={work_dir + out_dir + 'ActiveOctree.dat': actv})
    else:
        mesh.writeModelUBC(mesh, work_dir + out_dir + 'ActiveOctree.dat', actv)

    actvMap = Maps.InjectActiveCells(mesh, actv, 0)

    xyzLocs = survey.srcField.rxList[0].locs.copy()
    tiles = Utils.modelutils.tileSurveyPoints(xyzLocs, maxNpoints)

    X1, Y1 = tiles[0][:, 0], tiles[0][:, 1]
    X2, Y2 = tiles[1][:, 0], tiles[1][:, 1]


    # Plot data and tiles
#    fig, ax1 = plt.figure(), plt.subplot()
#    PF.Magnetics.plot_obs_2D(xyzLocs, survey.dobs, ax=ax1)
#    for ii in range(X1.shape[0]):
#        ax1.add_patch(Rectangle((X1[ii], Y1[ii]),
#                                X2[ii]-X1[ii],
#                                Y2[ii]-Y1[ii],
#                                facecolor='none', edgecolor='k'))
#    ax1.set_xlim([X1.min()-20, X2.max()+20])
#    ax1.set_ylim([Y1.min()-20, Y2.max()+20])
#    ax1.set_aspect('equal')
#    plt.show()
    
    # LOOP THROUGH TILES
    # expf = 1.3
    # dx = [mesh.hx.min(), mesh.hy.min()]
    surveyMask = np.ones(survey.nD, dtype='bool')
    # Going through all problems:
    # 1- Pair the survey and problem
    # 2- Add up sensitivity weights
    # 3- Add to the ComboMisfit

    nC = int(actv.sum())
    wrGlobal = np.zeros(nC)
    probSize = 0

    padDist = np.r_[np.c_[padLen, padLen], np.c_[padLen, padLen], np.c_[padLen, 0]]
    def createLocalProb(rxLoc, wrGlobal, lims):

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

        # mesh_t = meshTree.copy()
        mesh_t = Utils.modelutils.meshBuilder(rxLoc[ind_t, :], h, padDist,
                                            meshGlobal=meshInput,
                                            meshType='TREE',
                                            gridLoc='CC')

        mesh_t = Utils.modelutils.refineTree(mesh_t, topo, dtype='surface',
                                           nCpad=[0, 3, 2], finalize=False)

        mesh_t = Utils.modelutils.refineTree(mesh_t, rxLoc[ind_t, :], dtype='surface',
                                           nCpad=[10, 5, 5], finalize=False)

        center = np.mean(rxLoc[ind_t, :], axis=0)
        tileCenter = np.r_[np.mean(lims[0:2]), np.mean(lims[2:]), center[2]]

        ind = closestPoints(mesh, tileCenter, gridLoc='CC')

        shift = np.squeeze(mesh.gridCC[ind, :]) - center

        mesh_t.x0 += shift
        mesh_t.finalize()

        print(mesh_t.nC)
        actv_t = Utils.surface2ind_topo(mesh_t, topo)

        # Create reduced identity map
        tileMap = Maps.Tile((mesh, actv), (mesh_t, actv_t))
        tileMap.nCell = 40
        tileMap.nBlock = 1

        # Create the forward model operator
        prob = PF.Gravity.GravityIntegral(mesh_t, rhoMap=tileMap, actInd=actv_t,
                                           memory_saving_mode=True, parallelized=True)
        survey_t.pair(prob)

        # Data misfit function
        dmis = DataMisfit.l2_DataMisfit(survey_t)
        dmis.W = 1./survey_t.std

        wrGlobal += prob.getJtJdiag(np.ones(tileMap.P.shape[1]))

        # Create combo misfit function
        return dmis, wrGlobal

    for tt in range(X1.shape[0]):

        print("Tile " + str(tt+1) + " of " + str(X1.shape[0]))

        dmis, wrGlobal = createLocalProb(xyzLocs, wrGlobal, np.r_[X1[tt], X2[tt], Y1[tt], Y2[tt]])

        # Create combo misfit function

        if tt == 0:
            ComboMisfit = dmis

        else:
            ComboMisfit += dmis

        # Add problem size
    #    probSize += prob.F.shape[0] * prob.F.shape[1] * 32 / 4

    #ComboMisfit = ComboMisfit*1
    #print('Sum of all problems:' + str(probSize*1e-6) + ' Mb')
    # Scale global weights for regularization
    # Check if global mesh has regions untouched by local problem
    actvGlobal = wrGlobal != 0
    actvMeshGlobal = (wrGlobal[:nC]) != 0
    if actvMeshGlobal.sum() < actv.sum():

        for ind, dmis in enumerate(ComboMisfit.objfcts):
            dmis.prob.rhoMap.index = actvMeshGlobal
            dmis.prob.gtgdiag = None

    wrGlobal = wrGlobal[actvGlobal]**0.5
    wrGlobal = (wrGlobal/np.max(wrGlobal))

    #%% Create a regularization
    actv = np.all([actv, actvMap*actvMeshGlobal], axis=0)
    actvMap = Maps.InjectActiveCells(mesh, actv, 0)
    actvMapAmp = Maps.InjectActiveCells(mesh, actv, -100)

    nC = int(np.sum(actv))

    mstart = np.ones(nC)*1e-4
    mref = np.zeros(nC)

    # Create a regularization
    reg = Regularization.Sparse(mesh, indActive=actv, mapping=Maps.IdentityMap(nP=nC))
    reg.cell_weights = wrGlobal
    reg.norms = np.c_[driver.lpnorms].T
    reg.mref = mref

    # Add directives to the inversion
    opt = Optimization.ProjectedGNCG(maxIter=100, lower=-10., upper=10.,
                                     maxIterCG=20, tolCG=1e-3)

    invProb = InvProblem.BaseInvProblem(ComboMisfit, reg, opt)
    betaest = Directives.BetaEstimate_ByEig()

    # Here is where the norms are applied
    IRLS = Directives.Update_IRLS(f_min_change=1e-3,
                                  minGNiter=1)

    update_Jacobi = Directives.UpdateJacobiPrecond()
    targetMisfit = Directives.TargetMisfit()

    saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap)
    saveModel.fileName = work_dir + out_dir + 'GRAV'
    inv = Inversion.BaseInversion(invProb,
                                  directiveList=[betaest, IRLS, update_Jacobi,
                                                 saveModel])

    mrec = inv.run(mstart)

    if isinstance(mesh, Mesh.TreeMesh):
        Mesh.TreeMesh.writeUBC(
          mesh, work_dir + out_dir + 'OctreeMesh.msh',
          models={work_dir + out_dir + 'GRAV_Octree_l2.den': actvMap * invProb.l2model}
        )
    else:
        mesh.writeModelUBC(mesh, work_dir+out_dir + 'GRAV_l2.den', actvMap * invProb.l2model)

    # Get predicted data for each tile and write full predicted to file
    if getattr(ComboMisfit, 'objfcts', None) is not None:
        dpred = np.zeros(survey.nD)
        for ind, dmis in enumerate(ComboMisfit.objfcts):
            dpred[dmis.survey.ind] += dmis.survey.dpred(invProb.model)
    else:
        dpred = ComboMisfit.survey.dpred(invProb.model)

    PF.Gravity.writeUBCobs(
      work_dir+out_dir + 'GRAV_obs.pre', survey, dpred
    )

