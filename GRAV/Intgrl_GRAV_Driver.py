###############################################################################
# Gravity inversion
# -----------------
#
# This scripts runs a gravity inversion on a tensor mesh. It
#
#

from SimPEG import Mesh, Directives, Maps, InvProblem, Optimization, Utils
from SimPEG import DataMisfit, Inversion, Regularization
import SimPEG.PF as PF
import pylab as plt
import os
import numpy as np
#multiprocessing.freeze_support()

if __name__ == '__main__':

    #work_dir = 'C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Synthetic\\SingleBlock\\GRAV\\'
    work_dir = ".\\Tester\\"
#    work_dir = "C:\\Users\\DominiqueFournier\\Documents\\GIT\\InnovationGeothermal\\FORGE\\SyntheticModel\\"
    inpfile = 'SimPEG_GRAV.inp'
    out_dir = "SimPEG_GRAV_Inv\\"
    dsep = '\\'
    dsep = os.path.sep

    os.system('mkdir ' + work_dir + dsep + out_dir)


    # Read input file
    driver = PF.GravityDriver.GravityDriver_Inv(work_dir + dsep + inpfile)
    mesh = driver.mesh
    survey = driver.survey

    #nD = int(survey.dobs.shape[0]*0.1)
    #print("nD ratio:" + str(nD) +'\\' + str(survey.dobs.shape[0]) )
    #indx = np.random.randint(0, high=survey.dobs.shape[0], size=nD)
    ## Create a new downsampled survey
    #locXYZ = survey.srcField.rxList[0].locs[indx,:]
    #
    #dobs = survey.dobs
    #std = survey.std
    #
    #rxLoc = PF.BaseGrav.RxObs(locXYZ)
    #srcField = PF.BaseMag.SrcField([rxLoc], param=survey.srcField.param)
    #survey = PF.BaseMag.LinearSurvey(srcField)
    #survey.dobs = dobs[indx]
    #survey.std = std[indx]
    #
    #rxLoc = survey.srcField.rxList[0].locs
    #d = survey.dobs
    wd = survey.std

    ndata = survey.srcField.rxList[0].locs.shape[0]

    actv = driver.activeCells
    nC = len(actv)

    # Create active map to go from reduce set to full
    actvMap = Maps.InjectActiveCells(mesh, actv, -100)

    # Create static map
    static = driver.staticCells
    dynamic = driver.dynamicCells

    staticCells = Maps.InjectActiveCells(None, dynamic, driver.m0[static], nC=nC)
    mstart = driver.m0[dynamic]


    #%% Plot obs data
    # PF.Gravity.plot_obs_2D(survey.srcField.rxList[0].locs, survey.dobs,'Observed Data')

    #%% Run inversion
    prob = PF.Gravity.GravityIntegral(mesh=mesh, rhoMap=staticCells, actInd=actv,
                                      n_cpu=None, parallelized=True)
    prob.solverOpts['accuracyTol'] = 1e-4

    survey.pair(prob)

    dmis = DataMisfit.l2_DataMisfit(survey)
    dmis.W = 1./wd

    # Write out the predicted file and generate the forward operator
    pred = prob.fields(mstart)
    PF.Gravity.writeUBCobs(work_dir + dsep + out_dir + '\\Pred0.dat',survey,pred)

    # Load weighting  file
    if driver.wgtfile is None:
        # wr = PF.Magnetics.get_dist_wgt(mesh, rxLoc, actv, 3., np.min(mesh.hx)/4.)
        # wr = wr**2.

        # Make depth weighting
        wr = np.zeros(nC)
        for ii in range(survey.nD):
            wr += ((prob.F[ii, :]*prob.rhoMap.deriv(mstart)))**2.


        wr /= np.max(wr)
        wr **= 0.5
    #    wr /= mesh.vol[actv]**0.5
    #    wr = wr**0.5
        # wr_out = actvMap * wr

    else:
        wr = Mesh.TensorMesh.readModelUBC(mesh, work_dir + dsep + wgtfile)
        wr = wr[actv]
        wr = wr**2.

    # % Create inversion objects
    reg = Regularization.Sparse(mesh, indActive=actv, mapping=staticCells,gradientType='component')
    reg.mref = driver.mref[dynamic]
    reg.cell_weights = wr
    reg.norms = np.c_[driver.lpnorms].T
    if driver.eps is not None:
        reg.eps_p = driver.eps[0]
        reg.eps_q = driver.eps[1]


    opt = Optimization.ProjectedGNCG(maxIter=30,
                                     lower=driver.bounds[0],upper=driver.bounds[1],
                                     maxIterLS = 20, maxIterCG= 30, tolCG = 1e-4)

    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

    betaest = Directives.BetaEstimate_ByEig()
    IRLS = Directives.Update_IRLS(f_min_change=1e-4, minGNiter=2, chifact_start=1.)
    update_Jacobi = Directives.UpdateJacobiPrecond()
    saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap)
    saveModel.fileName = work_dir + dsep + out_dir + 'GRAV'

    saveDict = Directives.SaveOutputDictEveryIteration()
    inv = Inversion.BaseInversion(invProb, directiveList=[betaest, IRLS, saveDict,
                                                          update_Jacobi, saveModel])
    # Run inversion
    mrec = inv.run(mstart)

    # Plot predicted
    pred = prob.fields(mrec)

    # PF.Gravity.plot_obs_2D(survey, 'Observed Data')
    print("Final misfit:" + str(np.sum(((survey.dobs-pred)/wd)**2.)))

    m_out = actvMap*staticCells*invProb.l2model

    # Write result
    Mesh.TensorMesh.writeModelUBC(mesh, work_dir + dsep + out_dir + 'SimPEG_inv_l2l2.den', m_out)

    m_out = actvMap*staticCells*mrec
    # Write result
    Mesh.TensorMesh.writeModelUBC(mesh, work_dir + dsep + out_dir + 'SimPEG_inv_lplq.den', m_out)

    PF.Gravity.writeUBCobs(work_dir + out_dir + dsep + 'Predicted_l2.pre',
                             survey, d=survey.dpred(invProb.l2model))

    PF.Gravity.writeUBCobs(work_dir + out_dir + dsep + 'Predicted_lp.pre',
                             survey, d=survey.dpred(invProb.model))
