"""

This script runs an Magnetic Susceptibility Inversion from TMI data.

Created on December 7th, 2016

@author: fourndo@gmail.com

"""
from SimPEG import Mesh, Directives, Maps, InvProblem, Optimization, DataMisfit, Inversion, Utils, Regularization
import SimPEG.PF as PF
import numpy as np
import matplotlib.pyplot as plt
import os
from pymatsolver import PardisoSolver

if __name__ == '__main__':

    #work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Kevitsa\\Modeling\\MAG\\"
#    work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\\Research\\Synthetic\\Block_Gaussian_topo\\"
    #work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Synthetic\\SingleBlock\\Simpeg\\"
    #work_dir = "C:\\Egnyte\\Private\\dominiquef\\Projects\\4559_CuMtn_ZTEM\\Modeling\\MAG\\A1_Fenton\\"
    work_dir = "C:\\Users\\DominiqueFournier\\Dropbox\\Projects\\Synthetic\\Nut_Cracker\\"
    # work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\TKC\\DIGHEM_TMI\\"
#    work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Synthetic\\Triple_Block_lined\\"

    out_dir = "SimPEG_Susc_Inv\\"
    input_file = "SimPEG_MAG.inp"
    # %%
    # Read in the input file which included all parameters at once
    # (mesh, topo, model, survey, inv param, etc.)
    driver = PF.MagneticsDriver.MagneticsDriver_Inv(work_dir + input_file)

    os.system('if not exist ' + work_dir + out_dir + ' mkdir ' + work_dir+out_dir)

    # Access the mesh and survey information
    mesh = driver.mesh
    survey = driver.survey
    actv = driver.activeCells
    m0 = driver.m0 * np.random.randn(driver.m0.shape[0])  # Starting model

    nC = len(actv)


    # Create active map to go from reduce set to full
    actvMap = Maps.InjectActiveCells(mesh, actv, -100)

    # Creat reduced identity map
    idenMap = Maps.IdentityMap(nP=nC)

    # Create the forward model operator
    prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap,
                                         actInd=actv, parallelized=True)

    # Pair the survey and problem
    survey.pair(prob)

    # Create sensitivity weights from our linear forward operator
    rxLoc = survey.srcField.rxList[0].locs

    # Data misfit function
    dmis = DataMisfit.l2_DataMisfit(survey)
    dmis.W = 1./survey.std

    wr = prob.getJtJdiag(m0, W=dmis.W)

    wr = (wr/np.max(wr))
    wr = wr**0.5

    Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + 'SensWeights.sus',
                                  actvMap*wr)
    # wr = PF.Magnetics.get_dist_wgt(mesh, rxLoc, actv, 3, 1)

    # Create a regularization
    reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap, cell_weights = wr)
    reg.norms = driver.lpnorms.reshape((1,len(driver.lpnorms)))

    if driver.eps is not None:
        reg.eps_p = driver.eps[0]
        reg.eps_q = driver.eps[1]

    #reg.cell_weights = wr#driver.cell_weights*mesh.vol**0.5
    reg.mref = driver.mref


    # Add directives to the inversion
    opt = Optimization.ProjectedGNCG(maxIter=40, lower=0., upper=10.,
                                     maxIterLS=20, maxIterCG=10, tolCG=1e-4)
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    betaest = Directives.BetaEstimate_ByEig()

    # Apply Directives to the inversion
    IRLS = Directives.Update_IRLS(f_min_change=1e-3, minGNiter=1, maxIRLSiter=20)
    update_Jacobi = Directives.UpdateJacobiPrecond()
    saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap)
    saveModel.fileName = work_dir + out_dir + 'ModelSus'

    inv = Inversion.BaseInversion(invProb,
                                  directiveList=[betaest, saveModel,
                                                 IRLS, update_Jacobi])

    # Run the inversion
    mrec = inv.run(m0)

    Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + 'ModelSus_l2.sus',
                                  actvMap*invProb.l2model)

    Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + 'ModelSus_lp.sus',
                                  actvMap*invProb.model)

    PF.Magnetics.writeUBCobs(work_dir + out_dir + 'Predicted_l2.pre',
                             survey, d=survey.dpred(invProb.l2model))

    PF.Magnetics.writeUBCobs(work_dir + out_dir + 'Predicted_lp.pre',
                             survey, d=survey.dpred(invProb.model))
#%% Estimate JtJ and compare with wr
# m = m0
# k = 100
# if k is None:
#     k = int(survey.nD/10)

# def JtJv(v):

#     Jv = prob.Jvec(m, v)

#     return prob.Jtvec(m, Jv)

# JtJdiag = Utils.diagEst(JtJv, len(m), k=k,approach='PROBING' )

# #%%
# fig, ax = plt.subplots(1,1, figsize = (5, 5))
# dat1 = mesh.plotSlice(np.log(wr), ax = ax, normal = 'X')
# plt.colorbar(dat1[0], orientation="horizontal", ax = ax)
# ax.set_ylim(-500, 0)

# fig, ax = plt.subplots(1,1, figsize = (5, 5))
# dat1 = mesh.plotSlice(np.log(np.sum((prob.F)**2.,axis=0)), ax = ax, normal = 'X')
# plt.colorbar(dat1[0], orientation="horizontal", ax = ax)
# ax.set_ylim(-500, 0)
# plt.show()
