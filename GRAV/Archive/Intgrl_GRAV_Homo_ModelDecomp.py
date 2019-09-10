#%%
from SimPEG import Mesh, Directives, Maps, InvProblem, Optimization, Utils
from SimPEG import DataMisfit, Inversion, Regularization
import SimPEG.PF as PF
import pylab as plt
import os
import numpy as np

#work_dir = 'C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Synthetic\\SingleBlock\\GRAV\\'
#work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Synthetic\\Nut_Cracker\\"
#work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Synthetic\\Block_Gaussian_topo\\GRAV\\"
#work_dir = "C:\\Users\\DominiqueFournier\\Desktop\\Demo\\"
work_dir = "C:\\Users\\DominiqueFournier\\Dropbox\\Projects\\Synthetic\\Block_Gaussian_topo\\GRAV\\"
inpfile = 'SimPEG_GRAV.inp'
out_dir = "SimPEG_GRAV_ModelDecomp\\"
dsep = '\\'
dsep = os.path.sep

os.system('if not exist ' + work_dir + out_dir + ' mkdir ' + work_dir+out_dir)

# Choice for the homogeneous model
useMrefValues = True

# Read input file
driver = PF.GravityDriver.GravityDriver_Inv(work_dir + dsep + inpfile)
mesh = driver.mesh
survey = driver.survey
actv = driver.activeCells

# Tile the forward problem



m0 = driver.m0[actv]
mgeo = driver.mref[actv]

# Get unique geo units
geoUnits = np.unique(mgeo).tolist()

# Compute an a median value for each homogeneous units
mUnit = np.asarray([np.median(m0[mgeo==unit]) for unit in geoUnits])

# Create augmented mstart and mref
mref = np.r_[mUnit, mgeo]
mstart = np.r_[mUnit, m0]

# Build list of indecies for the geounits
index = []
for unit in geoUnits:
#    if unit!=0:
    index += [mgeo==unit]
nC = len(index)

# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

# Creat reduced identity map
homogMap = Maps.SurjectUnits(index)

# Create a wire map for a second model space
wires = Maps.Wires(('homo', nC), ('hetero', len(actv)))

# Create Sum map
sumMap = Maps.SumMap([homogMap*wires.homo, wires.hetero])


#%% Run inversion
prob = PF.Gravity.GravityIntegral(mesh, rhoMap=sumMap, actInd=actv)

survey.pair(prob)

#%%
# Load weighting  file

dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std

wr = prob.getJtJdiag(np.ones(int(nC + len(actv))), W=dmis.W)
wr[wires.homo.index] /= (np.max((wires.homo*wr)))
wr[wires.hetero.index] /= (np.max(wires.hetero*wr))
wr = wr**0.5


## Create a regularization
# For the homogeneous model
regMesh = Mesh.TensorMesh([nC])

reg_m1 = Regularization.Sparse(regMesh, mapping=wires.homo)
reg_m1.cell_weights = wires.homo*wr*2.
reg_m1.mref = mref

# Regularization for the voxel model
reg_m2 = Regularization.Sparse(mesh, indActive=actv, mapping=wires.hetero)
reg_m2.cell_weights = wires.hetero*wr
reg_m2.norms = np.c_[driver.lpnorms].T
reg_m2.mref =  mref

reg = reg_m1 + reg_m2

opt = Optimization.ProjectedGNCG(maxIter=30, lower=driver.bounds[0],
                                 upper=driver.bounds[1],
                                 maxIterLS = 20, maxIterCG= 30,
                                 tolCG = 1e-4)
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

betaest = Directives.BetaEstimate_ByEig(beta0_ratio = 1.)
IRLS = Directives.Update_IRLS(f_min_change=1e-4, minGNiter=2)
update_Jacobi = Directives.UpdatePreconditioner()
#saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap*sumMap)
#saveModel.fileName = work_dir + dsep + out_dir + 'GRAV'

saveDict = Directives.SaveOutputDictEveryIteration()
inv = Inversion.BaseInversion(invProb, directiveList=[betaest, IRLS, saveDict,
                                                      update_Jacobi])
# Run inversion
mrec = inv.run(mstart)

# Plot predicted
pred = prob.fields(mrec)

# PF.Gravity.plot_obs_2D(survey, 'Observed Data')
print("Final misfit:" + str(np.sum(((survey.dobs-pred)/survey.std)**2.)))

#%% Write result
if getattr(invProb, 'l2model', None) is not None:

    m_l2 = actvMap*(sumMap*invProb.l2model)
    Mesh.TensorMesh.writeModelUBC(mesh, work_dir + dsep + out_dir + 'Total_inv_l2l2.den', m_l2)

    m_l2 = actvMap*(homogMap*wires.homo*invProb.l2model)
    Mesh.TensorMesh.writeModelUBC(mesh, work_dir + dsep + out_dir + 'Homoge_inv_l2l2.den', m_l2)

    m_l2 = actvMap*(wires.hetero*invProb.l2model)
    Mesh.TensorMesh.writeModelUBC(mesh, work_dir + dsep + out_dir + 'Hetero_inv_l2l2.den', m_l2)

    PF.Gravity.writeUBCobs(work_dir + out_dir + dsep + 'Predicted_l2.pre',
                         survey, d=survey.dpred(invProb.l2model))

m_lp = actvMap*(sumMap*invProb.model)
Mesh.TensorMesh.writeModelUBC(mesh, work_dir + dsep + out_dir + 'Total_inv_lp.den', m_lp)

m_lp = actvMap*(homogMap*wires.homo*invProb.model)
Mesh.TensorMesh.writeModelUBC(mesh, work_dir + dsep + out_dir + 'Homoge_inv_lp.den', m_lp)

m_lp = actvMap*(wires.hetero*invProb.model)
Mesh.TensorMesh.writeModelUBC(mesh, work_dir + dsep + out_dir + 'Hetero_inv_lp.den', m_lp)

PF.Gravity.writeUBCobs(work_dir + out_dir + dsep + 'Predicted_lp.pre',
                         survey, d=invProb.dpred)
