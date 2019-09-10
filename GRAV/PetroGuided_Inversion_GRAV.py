from SimPEG import (
    Mesh,  Maps,  Utils, DataMisfit,  Regularization,
    Optimization, InvProblem,  Directives,  Inversion, PF
)
from SimPEG.EM.Static import DC, Utils as DCUtils
import numpy as np
import matplotlib.pyplot as plt
from pymatsolver import PardisoSolver
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import copy
import os

workDir = 'C:\\Users\\DominiqueFournier\\Dropbox\\Projects\\Synthetic\\Block_Gaussian_topo\\GRAV'
# workDir = "C:\\Users\\DominiqueFournier\\Desktop\\Workspace\\Paolo"
outDir = "SimPEG_GRAV_Petro_Inv\\"
inpFile = "SimPEG_GRAV.inp"
dsep = '\\'

os.system('mkdir ' + workDir + dsep + outDir)

# Load data, mesh and topography files
driver = PF.GravityDriver.GravityDriver_Inv(workDir + dsep + inpFile)
mesh = driver.mesh
survey = driver.survey
topo = driver.topo

# Read the geological model
geoModel = mesh.readModelUBC(workDir + dsep + "Block_sphere_model.den")

# Convert the unit to relative density
m0 = geoModel.copy()
# m0[m0==1] = 0.1

# Find the active cells (below topo)
actv = Utils.surface2ind_topo(mesh, topo, 'N')

m0 = m0[actv]
# #We can plot the data, looks like this,
# xyLocs = survey.srcField.rxList[0].locs
# fig = plt.figure(figsize=(8, 8))
# axs = plt.subplot()
# im, CS = Utils.PlotUtils.plotDataHillside(
#     xyLocs[:, 0], xyLocs[:, 1], survey.dobs, distMax=3500,
#     axs=axs, alpha=0.6,  alphaHS=1., altdeg=15,
#     ve=100)




# Choice for the homogeneous model
# useMrefValues = True

# Get unique geo units
# geoUnits = np.unique(geoModel).tolist()

# Compute an a median value for each homogeneous units
# mUnit = np.asarray([np.median(m0[geoModel==unit]) for unit in geoUnits])

# # Apply choice
# if useMrefValues:
#     mref = np.r_[mUnit, m0[actv]*0]
#     mstart = np.r_[mUnit, m0[actv]]
# else:
#     mref = np.r_[mUnit*0, m0[actv]*0]
#     mstart = np.r_[mUnit*0, m0[actv]]

#actv = mrho!=-100

# Build list of indecies for the geounits
# index = []
# for unit in geoUnits:
# #    if unit!=0:
#     index += [(geoModel==unit)[actv]]
# nC = len(index)

# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

nC = int(actv.sum())
# Creat reduced identity map
idenMap = Maps.IdentityMap(nP=nC)

# # Creat reduced identity map
# homogMap = Maps.SurjectUnits(index)
# homogMap.P

# # Create a wire map for a second model space
# wires = Maps.Wires(('homo', nC), ('hetero', int(actv.sum())))

# # Create Sum map
# sumMap = Maps.SumMap([homogMap*wires.homo, wires.hetero])


#%% Run inversion
prob = PF.Gravity.GravityIntegral(
    mesh, rhoMap=idenMap, actInd=actv,
    parallelized=True, Jpath=workDir + dsep + outDir + dsep + "sensitivity.zarr")

survey.pair(prob)


wr = prob.getJtJdiag(m0)
wr = wr**0.5
wr /= wr.max()



# ## Create a regularization
# # For the homogeneous model
# regMesh = Mesh.TensorMesh([nC])

# reg_m1 = Regularization.Sparse(regMesh, mapping=wires.homo)
# reg_m1.cell_weights = wires.homo*wr*100.
# reg_m1.norms = np.c_[2, 2, 2, 2]
# reg_m1.mref = mref

# # Regularization for the voxel model
# reg_m2 = Regularization.Sparse(mesh, indActive=actv, mapping=wires.hetero)
# reg_m2.cell_weights = wires.hetero*wr
# reg_m2.norms = np.c_[2, 2, 2, 2]
# reg_m2.mref =  mref

# reg = reg_m1 + reg_m2

dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std



# opt = Optimization.ProjectedGNCG(maxIter=3, lower=-.3,
#                                  upper=0.3,
#                                  maxIterLS = 20, maxIterCG= 30,
#                                  tolCG = 1e-4)

# invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=8e+2)

# betaest = Directives.BetaEstimate_ByEig(beta0_ratio = 1.)
# IRLS = Directives.Update_IRLS(f_min_change=1e-4, minGNiter=1, betaSearch=False)
# update_Jacobi = Directives.UpdatePreconditioner()
# #saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap*sumMap)
# #saveModel.fileName = workDir + dsep + out_dir + 'GRAV'

# saveDict = Directives.SaveOutputDictEveryIteration()
# inv = Inversion.BaseInversion(invProb, directiveList=[IRLS, saveDict,
#                                                       update_Jacobi])
# # Run inversion
# mrec = inv.run(mstart)


# m0 = np.median(ln_sigback) * np.ones(mapping.nP)
# dmis = DataMisfit.l2_DataMisfit(survey)

n = 3
clf = GaussianMixture(
    n_components=n,  covariance_type='full', reg_covar=5e-3
)

clf.fit(m0.reshape(-1, 1))
Utils.order_clusters_GM_weight(clf)
print(clf.covariances_)
print(clf.means_)
idenMap = Maps.IdentityMap(nP=m0.shape[0])
wires = Maps.Wires(('m', m0.shape[0]))
reg = Regularization.SimplePetroRegularization(
    GMmref=clf,  mesh=mesh,
    wiresmap=wires,
    maplist=[idenMap],
    mref=m0*0,
    indActive=actv,
    cell_weights=wr
)

reg.mrefInSmooth = False
reg.approx_gradient = True
gamma_petro = np.r_[1., 1., 1e-2]
reg.gamma = gamma_petro

opt = Optimization.ProjectedGNCG(
    maxIter=30, lower=-10, upper=10,
    maxIterLS=20, maxIterCG=50, tolCG=1e-4
)
opt.remember('xc')

invProb = InvProblem.BaseInvProblem(dmis,  reg,  opt)

Alphas = Directives.AlphasSmoothEstimate_ByEig(
    alpha0_ratio=1e-3, ninit=10, verbose=True
)
beta = Directives.BetaEstimate_ByEig(beta0_ratio=1e1)
betaIt = Directives.PetroBetaReWeighting(
    verbose=True, rateCooling=8.,
    rateWarming=1., tolerance=0.05
)
targets = Directives.PetroTargetMisfit(
    TriggerSmall=True,
    TriggerTheta=False,
    verbose=True,
)
MrefInSmooth = Directives.AddMrefInSmooth(verbose=True, wait_till_stable=True)
petrodir = Directives.GaussianMixtureUpdateModel()
#updateSensW = Directives.Update_DC_Wr(
#    wrType='sensitivityW',
#    changeMref=False, eps=1e-7
#)
# updateSensW = Directives.UpdateSensitivityWeights(threshold=1e-7)
update_Jacobi = Directives.UpdatePreconditioner()
saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap)
saveModel.fileName = workDir + dsep + outDir + 'GRAV_Petro'
inv = Inversion.BaseInversion(invProb,
                              directiveList=[Alphas, beta, saveModel,
                                             petrodir,
                                             targets, betaIt,
                                             MrefInSmooth,
                                             ])


mcluster = inv.run(m0**0*1e-4)

# Plot predicted
pred = prob.fields(mcluster)

# PF.Gravity.plot_obs_2D(survey, 'Observed Data')
print("Final misfit:" + str(np.sum(((survey.dobs-pred)/survey.std)**2.)))

#%% Write result
#if getattr(invProb, 'l2model', None) is not None:
#
#    m_l2 = actvMap*(sumMap*invProb.l2model)
#    Mesh.TensorMesh.writeModelUBC(mesh, workDir + dsep + outDir + 'Total_inv_l2l2.den', m_l2)
#
#    m_l2 = actvMap*(homogMap*wires.homo*invProb.l2model)
#    Mesh.TensorMesh.writeModelUBC(mesh, workDir + dsep + outDir + 'Homoge_inv_l2l2.den', m_l2)
#
#    m_l2 = actvMap*(wires.hetero*invProb.l2model)
#    Mesh.TensorMesh.writeModelUBC(mesh, workDir + dsep + outDir + 'Hetero_inv_l2l2.den', m_l2)
#
#    # PF.Gravity.writeUBCobs(workDir + outDir + dsep + 'Predicted_l2.pre',
#    #                      survey, d=survey.dpred(invProb.l2model))
#    Utils.io_utils.writeUBCgravityObservations(workDir + dsep + outDir + 'Predicted_l2.pre', survey, survey.dpred(invProb.l2model))
#
#m_lp = actvMap*(sumMap*invProb.model)
#Mesh.TensorMesh.writeModelUBC(mesh, workDir + dsep + outDir + 'Total_inv_lp.den', m_lp)
#
#m_lp = actvMap*(homogMap*wires.homo*invProb.model)
#Mesh.TensorMesh.writeModelUBC(mesh, workDir + dsep + outDir + 'Homoge_inv_lp.den', m_lp)
#
#m_lp = actvMap*(wires.hetero*invProb.model)
#Mesh.TensorMesh.writeModelUBC(mesh, workDir + dsep + outDir + 'Hetero_inv_lp.den', m_lp)

# PF.Gravity.writeUBCobs(workDir + outDir + dsep + 'Predicted_lp.pre',
#                          survey, d=invProb.dpred)
