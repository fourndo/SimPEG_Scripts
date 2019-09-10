"""
This script runs a Magnetic Vector Inversion - Spherical formulation. The code
is used to get the magnetization orientation and makes no induced assumption.

Created on Thu Sep 29 10:11:11 2016

@author: dominiquef
"""
from SimPEG import Mesh, Directives, Maps, InvProblem, Optimization
from SimPEG import DataMisfit, Inversion, Regularization, Utils
import SimPEG.PF as PF
import numpy as np
import os
import time

if __name__ == '__main__':
    # Define the inducing field parameter
    # work_dir = "C:\\Users\\DominiqueFournier\\Dropbox\\Projects\\Synthetic\\Nut_Cracker\\"
    # work_dir = "C:\\Users\\DominiqueFournier\\Dropbox\\Projects\\Synthetic\\Triple_Block_lined\\"
    work_dir = "C:\\Users\\DominiqueFournier\\Dropbox\\Projects\\Synthetic\\Block_Gaussian_topo\\"
    meshType = 'TreeMesh'
    #work_dir = "C:\\\Users\\DominiqueFournier\\Desktop\\Magnetics\\"
    out_dir = "SimPEG_MVIS\\"
    input_file = "SimPEG_MAG.inp"


    # %% INPUTS
    # Read in the input file which included all parameters at once
    # (mesh, topo, model, survey, inv param, etc.)
    driver = PF.MagneticsDriver.MagneticsDriver_Inv(work_dir + "\\" + input_file)

    os.system('mkdir ' + work_dir+out_dir)
    #%% Access the mesh and survey information
    meshInput = driver.mesh
    survey = driver.survey
    xyzLocs = survey.srcField.rxList[0].locs.copy()

    topo = None
    if driver.topofile is not None:
        topo = np.genfromtxt(driver.basePath + driver.topofile,
                             skip_header=1)
    else:
        # Grab the top coordinate and make a flat topo
        indTop = meshInput.gridCC[:,2] == meshInput.vectorCCz[-1]
        topo = meshInput.gridCC[indTop,:]
        topo[:,2] += meshInput.hz.min()/2. + 1e-8

    if meshType == 'TreeMesh':
        if isinstance(meshInput, Mesh.TensorMesh):
            # Define an octree mesh based on the provided tensor
            h = np.r_[meshInput.hx.min(), meshInput.hy.min(), meshInput.hz.min()]
            coreX, coreY, coreZ = meshInput.hx == h[0], meshInput.hy == h[1], meshInput.hz == h[2]
            padx, pady, padz = meshInput.hx[~coreX].sum(), meshInput.hy[~coreY].sum(), meshInput.hz[~coreZ].sum()

            padDist = np.r_[np.c_[padx, padx], np.c_[pady, pady], np.c_[padz, padz]]

            print("Creating TreeMesh. Please standby...")
            mesh = Utils.modelutils.meshBuilder(topo, h, padDist,
                                                meshGlobal=meshInput,
                                                meshType='TREE',
                                                verticalAlignment = 'center')

            mesh = Utils.modelutils.refineTree(mesh, topo, dtype='surface',
                                               nCpad=[0, 10, 2], finalize=False)

            mesh = Utils.modelutils.refineTree(mesh, xyzLocs, dtype='surface',
                                               nCpad=[10, 0, 0], finalize=True)

        else:
            mesh = Mesh.TreeMesh.readUBC(driver.basePath + driver.mshfile)
    else:
        mesh = meshInput

#    if driver.topofile is not None:
#        # Find the active cells
    actv = Utils.surface2ind_topo(mesh, topo)
#    else:
#        actv = mesh.gridCC[:, 2] <= meshInput.vectorNz[-1]
    if isinstance(mesh, Mesh.TreeMesh):
        Mesh.TreeMesh.writeUBC(mesh, work_dir + out_dir + 'OctreeMesh.msh',
                               models={work_dir + out_dir + 'ActiveOctree.dat': actv})
    else:
        mesh.writeModelUBC(work_dir + out_dir + 'ActiveTensor.dat', actv)
    #
    nC = int(actv.sum())
    ## Set starting mdoel
    mstart = np.ones(3*nC)*1e-4
    mref = np.zeros(3*nC)

    # Create active map to go from reduce space to full
    actvMap = Maps.InjectActiveCells(mesh, actv, 0)


    ## Create identity map
    idenMap = Maps.IdentityMap(nP=3*nC)

    # Create the forward model operator
    prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap, modelType='vector',
                                         actInd=actv, parallelized=True)

    # Explicitely set starting model
    prob.model = mstart

    # Pair the survey and problem
    survey.pair(prob)


    # RUN THE CARTESIAN FIRST TO GET A GOOD STARTING MODEL
    # Create sensitivity weights from our linear forward operator
    print('Calculating gtgdiag')
    dpre0 = prob.fields(mstart)
    wr = prob.getJtJdiag(mstart)**0.5
    wr = (wr/np.max(wr))

    # Create a block diagonal regularization
    wires = Maps.Wires(('p', nC), ('s', nC), ('t', nC))

    # Create a regularization
    reg_p = Regularization.Sparse(mesh, indActive=actv, mapping=wires.p)
    reg_p.cell_weights = (wires.p * wr)
    reg_p.norms = np.c_[2, 2, 2, 2]
    reg_p.mref = mref

    reg_s = Regularization.Sparse(mesh, indActive=actv, mapping=wires.s)
    reg_s.cell_weights = (wires.s * wr)
    reg_s.norms = np.c_[2, 2, 2, 2]
    reg_s.mref = mref

    reg_t = Regularization.Sparse(mesh, indActive=actv, mapping=wires.t)
    reg_t.cell_weights = (wires.t * wr)
    reg_t.norms = np.c_[2, 2, 2, 2]
    reg_t.mref = mref

    reg = reg_p + reg_s + reg_t
    reg.mref = mref

    # Data misfit function
    dmis = DataMisfit.l2_DataMisfit(survey)
    dmis.W = 1./survey.std

    # Add directives to the inversion
    opt = Optimization.ProjectedGNCG(maxIter=15, lower=-10., upper=10.,
                                     maxIterCG=20, tolCG=1e-3)

    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    betaest = Directives.BetaEstimate_ByEig()

    # This Directive controls the update for sparsity
    IRLS = Directives.Update_IRLS(f_min_change=1e-3,
                                  minGNiter=1)

    update_Jacobi = Directives.UpdatePreconditioner()
    targetMisfit = Directives.TargetMisfit()

    saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap, vector=True)
    saveModel.fileName = work_dir + out_dir + 'MVI_C'
    inv = Inversion.BaseInversion(invProb,
                                  directiveList=[betaest, IRLS, update_Jacobi,
                                                 saveModel])

    mrec_MVI = inv.run(mstart)

    x = actvMap * (wires.p * mrec_MVI)
    y = actvMap * (wires.s * mrec_MVI)
    z = actvMap * (wires.t * mrec_MVI)

    amp =  (np.sum(np.c_[x, y, z]**2., axis=1))**0.5

    # if isinstance(mesh, Mesh.TreeMesh):
    #     Mesh.TreeMesh.writeUBC(mesh, work_dir + out_dir + 'OctreeMesh.msh',
    #                        models={work_dir + out_dir + 'MVI_C_amp.sus': amp})
    # else:
    #     mesh.writeModelUBC(work_dir+out_dir + 'MVI_C_amp.sus', amp)

    Utils.io_utils.writeUBCmagneticsObservations(work_dir+out_dir + 'MVI_C_pred.pre', survey, invProb.dpred)

    #mstart = PF.Magnetics.readVectorModel(mesh, work_dir + 'mviinv_019.fld')
    #mstart = Utils.matutils.xyz2atp(mstart.reshape((nC,3),order='F'))
    beta = invProb.beta
    #beta = 3.415026E+06
    # %% RUN MVI-S WITH SPARSITY

    # # STEP 3: Finish inversion with spherical formulation
    mstart = Utils.matutils.xyz2atp(mrec_MVI.reshape((nC,3),order='F'))

    prob.coordinate_system = 'spherical'
    prob.model = mstart

    dd = prob.fields(mstart)
    # Create a block diagonal regularization
    wires = Maps.Wires(('amp', nC), ('theta', nC), ('phi', nC))

    # Create a regularization
    reg_a = Regularization.Sparse(mesh, indActive=actv, mapping=wires.amp)

    reg_a.norms = np.c_[driver.lpnorms[:4]].T
    if driver.eps is not None:
        reg_a.eps_p = driver.eps[0]
        reg_a.eps_q = driver.eps[1]

    reg_a.mref = mref

    reg_t = Regularization.Sparse(mesh, indActive=actv, mapping=wires.theta)
    reg_t.alpha_s = 0.  # No reference angle
    reg_t.space = 'spherical'
    reg_t.norms = np.c_[driver.lpnorms[4:8]].T
    reg_t.eps_q = 5e-2
    reg_t.mref = mref
    # reg_t.alpha_x, reg_t.alpha_y, reg_t.alpha_z = 0.25, 0.25, 0.25

    reg_p = Regularization.Sparse(mesh, indActive=actv, mapping=wires.phi)
    reg_p.alpha_s = 0.  # No reference angle
    reg_p.space = 'spherical'
    reg_p.norms = np.c_[driver.lpnorms[8:]].T
    reg_p.eps_q = 5e-2
    reg_p.mref = mref

    reg = reg_a + reg_t + reg_p
    reg.mref = mref

    # Data misfit function
    dmis = DataMisfit.l2_DataMisfit(survey)
    dmis.W = 1./survey.std

    Lbound = np.kron(np.asarray([0, -np.inf, -np.inf]), np.ones(nC))
    Ubound = np.kron(np.asarray([10, np.inf, np.inf]), np.ones(nC))


    # Add directives to the inversion
    opt = Optimization.ProjectedGNCG(maxIter=20,
                                     lower=Lbound,
                                     upper=Ubound,
                                     maxIterLS=10,
                                     maxIterCG=20, tolCG=1e-3,
                                     stepOffBoundsFact=1e-8)

    #invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=beta*2)
    #  betaest = Directives.BetaEstimate_ByEig()

    # Here is where the norms are applied
    IRLS = Directives.Update_IRLS(f_min_change=1e-4, maxIRLSiter=20,
                                  minGNiter=1, beta_tol = 0.5, prctile=100,
                                  coolingRate=1, coolEps_q=True,
                                  betaSearch=True)

    invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=beta)

    # Special directive specific to the mag amplitude problem. The sensitivity
    # weights are update between each iteration.
    ProjSpherical = Directives.ProjSpherical()
    update_SensWeight = Directives.UpdateSensitivityWeights()
    update_Jacobi = Directives.UpdatePreconditioner()
    saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap, vector=True)
    saveModel.fileName = work_dir+out_dir + 'MVI_S'

    inv = Inversion.BaseInversion(invProb,
                                  directiveList=[ProjSpherical, IRLS,
                                                 update_SensWeight,
                                                 update_Jacobi, saveModel])

    tstart = time.time()
    mrec_MVI_S = inv.run(mstart)
    #%%
    print('Total runtime: ' + str(time.time()-tstart))

    # if isinstance(mesh, Mesh.TreeMesh):
    #     Mesh.TreeMesh.writeUBC(
    #             mesh, work_dir + out_dir + 'OctreeMesh.msh',
    #             models={
    #                 work_dir + out_dir + 'MVI_S_amp.sus': actvMap * (mrec_MVI_S[:nC])
    #             }
    #     )
    #     Mesh.TreeMesh.writeUBC(
    #             mesh, work_dir + out_dir + 'OctreeMesh.msh',
    #             models={
    #                 work_dir + out_dir + 'MVI_S_theta.sus': actvMap * (mrec_MVI_S[nC:2*nC])
    #             }
    #     )
    #     Mesh.TreeMesh.writeUBC(
    #             mesh, work_dir + out_dir + 'OctreeMesh.msh',
    #             models={
    #                 work_dir + out_dir + 'MVI_S_phi.sus': actvMap * (mrec_MVI_S[2*nC:])
    #             }
    #     )
    # else:
    #     Mesh.TensorMesh.writeModelUBC(mesh, work_dir+out_dir + 'MVI_S_amp.sus',
    #                                   actvMap * (mrec_MVI_S[:nC]))
    #     Mesh.TensorMesh.writeModelUBC(mesh, work_dir+out_dir + 'MVI_S_theta.sus',
    #                                   actvMap * (mrec_MVI_S[nC:2*nC]))
    #     Mesh.TensorMesh.writeModelUBC(mesh, work_dir+out_dir + 'MVI_S_phi.sus',
    #                                   actvMap * (mrec_MVI_S[2*nC:]))

    #     Mesh.TensorMesh.writeModelUBC(mesh, work_dir+out_dir + 'MVI_S_l2_amp.sus',
    #                                   actvMap * (invProb.l2model[:nC]))
    #     Mesh.TensorMesh.writeModelUBC(mesh, work_dir+out_dir + 'MVI_S_l2_theta.sus',
    #                                   actvMap * (invProb.l2model[nC:2*nC]))
    #     Mesh.TensorMesh.writeModelUBC(mesh, work_dir+out_dir + 'MVI_S_l2_phi.sus',
    #                                   actvMap * (invProb.l2model[2*nC:]))

    #     vec_xyz = Utils.matutils.atp2xyz(invProb.l2model.reshape((nC, 3), order='F'))

    #     vec_x = actvMap * vec_xyz[:nC]
    #     vec_y = actvMap * vec_xyz[nC:2*nC]
    #     vec_z = actvMap * vec_xyz[2*nC:]

    #     vec = np.c_[vec_x, vec_y, vec_z]

    #     PF.MagneticsDriver.writeVectorUBC(mesh,
    #                                       work_dir+out_dir + 'MVI_S_l2_VEC.fld', vec)

    Utils.io_utils.writeUBCmagneticsObservations(work_dir+out_dir + 'MVI_S_pred.pre', survey, invProb.dpred)
