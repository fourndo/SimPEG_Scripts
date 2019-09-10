"""
This script runs a Magnetic Vector Inversion - Spherical formulation. The code
is used to get the magnetization orientation and makes no induced assumption.

Created on Thu Sep 29 10:11:11 2016

@author: dominiquef
"""
from SimPEG import Mesh, Directives, Maps, InvProblem, Optimization, Utils
from SimPEG import DataMisfit, Inversion, Regularization
import SimPEG.PF as PF
import numpy as np
import os

# Define the inducing field parameter
# work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\\Research\\Modelling\\Synthetic\\Block_Gaussian_topo\\"
#work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Modelling\\Synthetic\\Triple_Block_lined\\"
work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Synthetic\\Nut_Cracker\\"

out_dir = "SimPEG_JOINT_Inv\\"
input_file = "SimPEG_MAG.inp"


# %% INPUTS
# Read in the input file which included all parameters at once
# (mesh, topo, model, survey, inv param, etc.)
driver = PF.MagneticsDriver.MagneticsDriver_Inv(work_dir + input_file)
os.system('if not exist ' +work_dir+out_dir + ' mkdir ' + work_dir+out_dir)
#%% Access the mesh and survey information
mesh = driver.mesh
survey = driver.survey

# Get the active cells for equivalent source is the top only
active = driver.activeCells
surf = PF.MagneticsDriver.actIndFull2layer(mesh, active)

# Get the layer of cells directyl below topo
#surf = Utils.actIndFull2layer(mesh, active)
nC = len(surf)  # Number of active cells

# Create active map to go from reduce set to full
surfMap = Maps.InjectActiveCells(mesh, surf, -100)

# Create identity map
idenMap = Maps.IdentityMap(nP=nC)

# Create static map
prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap, actInd=surf, equiSourceLayer=True)
prob.solverOpts['accuracyTol'] = 1e-4

# Pair the survey and problem
survey.pair(prob)

# Create a regularization function, in this case l2l2
reg = Regularization.Simple(mesh, indActive=surf)
reg.mref = np.zeros(nC)

# Specify how the optimization will proceed, set susceptibility bounds to inf
opt = Optimization.ProjectedGNCG(maxIter=25, lower=-np.inf,
                                 upper=np.inf, maxIterLS=20,
                                 maxIterCG=20, tolCG=1e-3)

# Define misfit function (obs-calc)
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std

# Create the default L2 inverse problem from the above objects
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

# Specify how the initial beta is found
betaest = Directives.BetaEstimate_ByEig()

# Beta schedule for inversion
betaSchedule = Directives.BetaSchedule(coolingFactor=2., coolingRate=1)

# Target misfit to stop the inversion,
# try to fit as much as possible of the signal, we don't want to lose anything
targetMisfit = Directives.TargetMisfit(chifact=0.1)

# Put all the parts together
inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, betaSchedule, targetMisfit])

# Run the equivalent source inversion
mstart = np.zeros(nC)
mrec = inv.run(mstart)

# %% STEP 2: COMPUTE AMPLITUDE DATA
# Now that we have an equialent source layer, we can forward model alh three
# components of the field and add them up: |B| = ( Bx**2 + Bx**2 + Bx**2 )**0.5

# Won't store the sensitivity and output 'xyz' data.
prob.forwardOnly = True
pred_x = prob.Intrgl_Fwr_Op(m=mrec, recType='x')
pred_y = prob.Intrgl_Fwr_Op(m=mrec, recType='y')
pred_z = prob.Intrgl_Fwr_Op(m=mrec, recType='z')

ndata = survey.nD

d_amp = np.sqrt(pred_x**2. +
                pred_y**2. +
                pred_z**2.)


Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + "EquivalentSource.sus", surfMap*mrec)

rxLoc = PF.BaseMag.RxObs(survey.srcField.rxList[0].locs)
srcField = PF.BaseMag.SrcField([rxLoc], param= survey.srcField.param)
survey_amp = PF.BaseMag.LinearSurvey(srcField)

# Change the survey to xyz components
survey_amp.srcField.rxList[0].rxType = 'xyz'
survey_amp.dobs = d_amp
survey_amp.std = survey.std
# %% Create MIV problem 
# Extract active region
actv = driver.activeCells
actvMap = Maps.InjectActiveCells(mesh, actv, 0)
nC = actv.shape[0]
# # RUN THE MVI-CARTESIAN FIRST
# Create identity map
idenMap = Maps.IdentityMap(nP=3*nC)

# Create the forward model operator
prob_MVI = PF.Magnetics.MagneticVector(mesh, chiMap=idenMap,
                                     actInd=actv)

# Explicitely set starting model
#prob_MVI.model = mstart
#
## Pair the survey and problem
survey.pair(prob_MVI)
#
#
# Create sensitivity weights from our linear forward operator
wr = np.sum(prob_MVI.F**2., axis=0)**0.5
wr = (wr/np.max(wr))

# Create a block diagonal regularization
wires = Maps.Wires(('p', nC), ('s', nC), ('t', nC))

# Create a regularization
reg_p = Regularization.Sparse(mesh, indActive=actv, mapping=wires.p)
reg_p.cell_weights = wires.p * wr
reg_p.norms = [2, 2, 2, 2]

reg_s = Regularization.Sparse(mesh, indActive=actv, mapping=wires.s)
reg_s.cell_weights = wires.s * wr
reg_s.norms = [2, 2, 2, 2]

reg_t = Regularization.Sparse(mesh, indActive=actv, mapping=wires.t)
reg_t.cell_weights = wires.t * wr
reg_t.norms = [2, 2, 2, 2]

reg = reg_p + reg_s + reg_t
reg.mref = np.zeros(3*nC)

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=7, lower=-10., upper=10.,
                                 maxIterCG=20, tolCG=1e-3)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
betaest = Directives.BetaEstimate_ByEig()

# Here is where the norms are applied
IRLS = Directives.Update_IRLS(f_min_change=1e-4,
                              minGNiter=3, beta_tol=1e-2)

update_Jacobi = Directives.UpdatePreCond()
targetMisfit = Directives.TargetMisfit()
saveModel = Directives.SaveUBCModelEveryIteration(mapping = actvMap)
saveModel.fileName = work_dir+out_dir + 'MVI_C'

inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, IRLS, update_Jacobi,
                                             saveModel])

mrec_MVI = inv.run(np.ones(3*nC)*1e-4)

beta = invProb.beta

#%% # Now create the joint problems

# AMPLITUDE
# Create active map to go from reduce space to full
#M = PF.Magnetics.dipazm_2_xyz(np.ones(nC) * survey.srcField.param[1], np.ones(nC) * survey.srcField.param[2])


mstart = PF.Magnetics.xyz2atp(mrec_MVI)

mcol = mrec_MVI.reshape((nC, 3), order='F')
amp = np.sum(mcol**2.,axis=1)**0.5
M = Utils.sdiag(1./amp) * mcol

# Create identity map
idenMap = Maps.IdentityMap(nP=3*nC)
#
wires = Maps.Wires(('p', nC), ('s', nC), ('t', nC))

# Create the forward model operator
prob_amp = PF.Magnetics.MagneticAmplitude(mesh, chiMap=wires.p,
                                      actInd=actv,
                                      silent=True)


# Pair the survey and problem
survey_amp.pair(prob_amp)


# Data misfit function
dmis_amp = DataMisfit.l2_DataMisfit(survey_amp)
dmis_amp.W = 1./survey_amp.std


#  MVI-S

#mstart = PF.Magnetics.xyz2atp(mrec_MVI)
prob_MVI.coordinate_system = 'spherical'

prob_MVI.model = mstart
# Create a block diagonal regularization
# Create a block diagonal regularization
wires = Maps.Wires(('amp', nC), ('theta', nC), ('phi', nC))

# Create a regularization
reg_a = Regularization.Sparse(mesh, indActive=actv, mapping=wires.amp)
reg_a.norms = driver.lpnorms[:4]
if driver.eps is not None:
    reg_a.eps_p = driver.eps[0]
    reg_a.eps_q = driver.eps[1]
else:
    reg_a.eps_p = np.percentile(np.abs(mstart[:nC]), 95)

reg_t = Regularization.Sparse(mesh, indActive=actv, mapping=wires.theta)
reg_t.alpha_s = 0.
reg_t.space = 'spherical'
reg_t.norms = [2,0,0,0]
reg_t.eps_q = 2e-2
# reg_t.alpha_x, reg_t.alpha_y, reg_t.alpha_z = 0.25, 0.25, 0.25

reg_p = Regularization.Sparse(mesh, indActive=actv, mapping=wires.phi)
reg_p.alpha_s = 0.
reg_p.space = 'spherical'
reg_p.norms = [2,0,0,0]
reg_p.eps_q = 1e-2

reg = reg_a + reg_t + reg_p
reg.mref = np.zeros(3*nC)

# Data misfit function
dmis_MVI = DataMisfit.l2_DataMisfit(survey)
dmis_MVI.W = 1./survey.std

# JOIN TO PROBLEMS
dmis = dmis_amp + dmis_MVI

Lbound = np.kron(np.asarray([0, -np.inf, -np.inf]),np.ones(nC))
Ubound = np.kron(np.asarray([10, np.inf, np.inf]),np.ones(nC))

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=30,
                                 lower=Lbound,
                                 upper=Ubound,
                                 maxIterLS=10,
                                 maxIterCG=30, tolCG=1e-4,
                                 LSalwaysPass=True,
                                 stepOffBoundsFact=1e-8)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

# LIST OF DIRECTIVES
# betaest = Directives.BetaEstimate_ByEig()
IRLS = Directives.Update_IRLS(f_min_change=1e-4,
                              minGNiter=2, beta_tol=1e-2,
                              coolingRate=2)
update_SensWeight = Directives.UpdateSensWeighting()
update_Jacobi = Directives.UpdatePreCond(epsilon=1e-7)
ProjSpherical = Directives.ProjSpherical()
JointAmpMVI = Directives.JointAmpMVI()
betaest = Directives.BetaEstimate_ByEig(beta0_ratio = 1e+1)
saveModel = Directives.SaveUBCModelEveryIteration(mapping = actvMap)
saveModel.fileName = work_dir+out_dir + 'JOINT_MVIS_A'

inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, ProjSpherical, JointAmpMVI, IRLS, update_SensWeight, 
                                             update_Jacobi, saveModel])

# Run JOINT

mrec = inv.run(mstart)

#%%
mvec = PF.Magnetics.atp2xyz(mrec)
PF.Magnetics.plotModelSections(mesh, mvec, normal='z',
                               ind=-5,
                               scale=0.1, vec='w',
                               )
