"""
This script runs a Magnetic Vector Inversion - Cartesian formulation. The code
is used to get the magnetization orientation and makes no induced assumption.
The inverse problem is three times larger than the usual susceptibility
inversion, but won't suffer the same issues in the presence of remanence. The
drawback of having 3x the number of model parameters is in non-uniqueness of
the solution. Models have a tendency to be overlly complicated.

To counter this issue, the inversion can be done cooperatively with the
Magnetic Amplitude Inverion (MAI) that uses sparse norms to recover compact
bodies. The information about the location and shape of magnetic bodies is
used by the MVI code through a cell-based wieght.

This script will run both inversions in sequence in order to compare the
results if the flag CMI is activated

-------------------------------------------------------------------------------
 !! IMPORTANT: PLease run Intgrl_MAG_AMP_Inv.py before running this script !!
-------------------------------------------------------------------------------


Created on Thu Sep 29 10:11:11 2016

@author: dominiquef
"""
from SimPEG import Mesh, Directives, Maps, InvProblem, Optimization
from SimPEG import DataMisfit, Inversion, Regularization, Utils
import SimPEG.PF as PF
import numpy as np
import os

# Define the inducing field parameter
# work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\\Research\\Modelling\\Synthetic\\Block_Gaussian_topo\\"
#work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Modelling\\Synthetic\\SingleBlock\\Simpeg\\"
work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Modelling\\Synthetic\\Nut_Cracker\\"
out_dir = "SimPEG_PF_Inv\\"
input_file = "SimPEG_MAG.inp"


# %% INPUTS
# Read in the input file which included all parameters at once
# (mesh, topo, model, survey, inv param, etc.)
driver = PF.MagneticsDriver.MagneticsDriver_Inv(work_dir + input_file)
os.system('if not exist ' +work_dir+out_dir + ' mkdir ' + work_dir+out_dir)
#%% Access the mesh and survey information
mesh = driver.mesh
survey = driver.survey

# %% STEP 1: EQUIVALENT SOURCE LAYER
# The first step inverts for an equiavlent source layer in order to convert the
# observed TMI data to magnetic field Amplitude.

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

# Ouput result
Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + "EquivalentSource.sus", surfMap*mrec)


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

## RUN JOINT INVERSION
survey_amp = PF.BaseMag.LinearSurvey(survey.srcField)
survey_amp.dobs = d_amp
survey_amp.std = survey.std

B = survey.srcField.param



# Extract active region
actv = driver.activeCells

# Create active map to go from reduce space to full
actvMap = Maps.InjectActiveCells(mesh, actv, 0)
nC = int(len(actv))

M = PF.Magnetics.dipazm_2_xyz(np.ones(nC) * B[1], np.ones(nC) * B[2])

mstart = Utils.mkvc(M)*1e-4

idenMap = Maps.IdentityMap(nP=3*nC)

# Create a block diagonal misfit wire

# Create the forward model operator
prob_MVI = PF.Magnetics.MagneticVector(mesh, chiMap=idenMap,
                                     actInd=actv, silent=True)

# Explicitely set starting model
prob_MVI.model = mstart

# Pair the survey and problem
survey.pair(prob_MVI)

# Create sensitivity weights from our linear forward operator
wr = np.sum(prob_MVI.F**2., axis=0)

#prob_MVI.JtJdiag = wr

dmis_MVI = DataMisfit.l2_DataMisfit(survey)
dmis_MVI.W = 1./survey.std



#%% # Now create the joint problems

# AMPLITUDE
# Create active map to go from reduce space to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)
nC = int(len(actv))



# Create the forward model operator
prob_amp = PF.Magnetics.MagneticAmplitude(mesh, chiMap=idenMap,
                                      actInd=actv, M=M, coordinate_system='cartesian',
                                      silent=True)

prob_amp.model = mstart
# Change the survey to xyz components
survey_amp.srcField.rxList[0].rxType = 'xyz'

# Pair the survey and problem
survey_amp.pair(prob_amp)

# Data misfit function
dmis_amp = DataMisfit.l2_DataMisfit(survey_amp)
dmis_amp.W = 1./survey_amp.std


# Create a block diagonal regularization
wires = Maps.Wires(('p', nC), ('s', nC), ('t', nC))

# Create an amplitude map
ampMap = Maps.AmplitudeMap(mesh)

# Create a regularization
reg_a = Regularization.Sparse(mesh, indActive=actv, mapping=ampMap)
reg_a.norms = [0, 1, 1, 1]
reg_a.alpha_s = 1e+2
reg_a.alpha_x = 1e+2
reg_a.alpha_y = 1e+2
reg_a.alpha_z = 1e+2
#reg_a.eps_p = 5e-3
#reg_a.eps_q = 5e-3

# Create a regularization
reg_p = Regularization.Sparse(mesh, indActive=actv, mapping=wires.p)
reg_p.norms = [2, 2, 2, 2]
reg_p.eps_p = 1e-3
reg_p.eps_q = 1e-3


reg_s = Regularization.Sparse(mesh, indActive=actv, mapping=wires.s)
reg_s.norms = [2, 2, 2, 2]
reg_s.eps_p = 1e-3
reg_s.eps_q = 1e-3

reg_t = Regularization.Sparse(mesh, indActive=actv, mapping=wires.t)
reg_t.norms = [2, 2, 2, 2]
reg_t.eps_p = 1e-3
reg_t.eps_q = 1e-3

reg = reg_a + reg_p + reg_s + reg_t
reg.mref = np.zeros(3*nC)


# JOIN TO PROBLEMS
dmis = dmis_amp + dmis_MVI

Lbound = np.kron(np.asarray([-np.inf, -np.inf, -np.inf]),np.ones(nC))
Ubound = np.kron(np.asarray([np.inf, np.inf, np.inf]),np.ones(nC))

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=20,
                                 lower=Lbound,
                                 upper=Ubound,
                                 maxIterLS=10,
                                 maxIterCG=20, tolCG=1e-3,
                                 LSalwaysPass=True,
                                 stepOffBoundsFact=1e-8)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt)


# LIST OF DIRECTIVES
# betaest = Directives.BetaEstimate_ByEig()
IRLS = Directives.Update_IRLS(f_min_change=1e-6,
                              minGNiter=2, beta_tol=1e-2,
                              coolingRate=2)
update_SensWeight = Directives.UpdateSensWeighting()
update_Jacobi = Directives.UpdatePreCond()
ProjSpherical = Directives.ProjSpherical()
JointAmpMVI = Directives.JointAmpMVI()
betaest = Directives.BetaEstimate_ByEig(beta0_ratio = 1e+3)
saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap,
                                                    saveComp=True)
saveModel.fileName = work_dir+out_dir + 'JOINT_MVIC_A'

inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, IRLS, update_SensWeight,
                                             update_Jacobi, saveModel])

# Run JOINT
mrec = inv.run(mstart)

#NOTE - Would like to have dpred working on both surveys
dpred = invProb.getFields(mrec)

print('Amplitude Final phi_d: ' + str(dmis_amp(mrec)))
print('MVI Final phi_d: ' + str(dmis_MVI(mrec)))

Mesh.TensorMesh.writeModelUBC(mesh,work_dir+out_dir + 'JOINT_MVIC_A' + '.sus',ampMap*mrec)
PF.MagneticsDriver.writeVectorUBC(mesh,work_dir+out_dir + 'JOINT_MVIC_A' + '.fld',(wires_misfit.pst*mrec).reshape((nC,3), order='f'))