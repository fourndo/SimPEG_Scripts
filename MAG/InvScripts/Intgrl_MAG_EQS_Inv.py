"""

This script runs an equivalent source (ES) layer from TMI data.
The same layer is used to forwward x,y,z and amplitude field data.

Created on January 16th, 2019

@author: fourndo@gmail.com

"""
from SimPEG import Mesh, Directives, Maps, InvProblem, Optimization, DataMisfit, Inversion, Utils, Regularization
import SimPEG.PF as PF
import numpy as np
import matplotlib.pyplot as plt
import os

#work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Modelling\\Synthetic\\Triple_Block_lined\\"
#work_dir = "C:\\Users\\DominiqueFournier\\Dropbox\\Projects\\Synthetic\\Nut_Cracker\\"
work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Osborne\\Inversion\\UTM\\"
# work_dir = "C:\\Users\\DominiqueFournier\\Downloads\\Mages_01\\Mages_01\\"
# work_dir = "C:\\Users\\DominiqueFournier\\Documents\\GIT\\InnovationGeothermal\\FORGE\\"
# work_dir = "C:\\Users\\DominiqueFournier\\Documents\\GIT\\InnovationGeothermal\\"
out_dir = "SimPEG_MAG_ES_Inv\\"
input_file = "SimPEG_MAG.inp"
# %%
# Read in the input file which included all parameters at once
# (mesh, topo, model, survey, inv param, etc.)
driver = PF.MagneticsDriver.MagneticsDriver_Inv(work_dir + input_file)

os.system('if not exist ' + work_dir + out_dir + ' mkdir ' + work_dir+out_dir)

# Access the mesh and survey information
mesh = driver.mesh
survey = driver.survey

# %% STEP 1: EQUIVALENT SOURCE LAYER
# The first step inverts for an equiavlent source layer in order to convert the
# observed TMI data to magnetic field Amplitude.

# Get the active cells for equivalent source is the top only
active = driver.activeCells
surf = PF.MagneticsDriver.actIndFull2layer(mesh, active)

nC = len(surf)  # Number of active cells

# Create active map to go from reduce set to full
surfMap = Maps.InjectActiveCells(mesh, surf, -100)

# Create identity map
idenMap = Maps.IdentityMap(nP=nC)

# Create static map
prob = PF.Magnetics.MagneticIntegral(
    mesh, chiMap=idenMap, actInd=surf,
    parallelized=True,
    Jpath=work_dir + out_dir + "Sensitivity.zarr",
    equiSourceLayer=True)
prob.solverOpts['accuracyTol'] = 1e-4

# Pair the survey and problem
survey.pair(prob)

# Define misfit function (obs-calc)
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std

wr = prob.getJtJdiag(np.ones(nC), W=dmis.W)
wr = (wr/np.max(wr))
wr = wr**0.5

# Create a regularization function, in this case l2l2
reg = Regularization.Sparse(
    mesh, indActive=surf, mapping=Maps.IdentityMap(nP=nC)
)
reg.mref = np.zeros(nC)
reg.cell_weights = wr

# Specify how the optimization will proceed, set susceptibility bounds to inf
opt = Optimization.ProjectedGNCG(maxIter=25, lower=-np.inf,
                                 upper=np.inf, maxIterLS=20,
                                 maxIterCG=20, tolCG=1e-3)

# Create the default L2 inverse problem from the above objects
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

# Specify how the initial beta is found
betaest = Directives.BetaEstimate_ByEig(beta0_ratio = 1e-2)

# Beta schedule for inversion
betaSchedule = Directives.BetaSchedule(coolingFactor=2., coolingRate=1)

# Target misfit to stop the inversion,
# try to fit as much as possible of the signal, we don't want to lose anything
targetMisfit = Directives.TargetMisfit(chifact=0.1)

saveModel = Directives.SaveUBCModelEveryIteration(mapping=surfMap)
saveModel.fileName = work_dir + out_dir + 'EquivalentSource'

# Put all the parts together
inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, betaSchedule, saveModel, targetMisfit])

# Run the equivalent source inversion
mstart = np.zeros(nC)
mrec = inv.run(mstart)


# %% STEP 2: COMPUTE AMPLITUDE DATA
# Now that we have an equialent source layer, we can forward model alh three
# components of the field and add them up: |B| = ( Bx**2 + Bx**2 + Bx**2 )**0.5

# Won't store the sensitivity and output 'xyz' data.
prob = PF.Magnetics.MagneticIntegral(
    mesh, chiMap=idenMap, actInd=surf,
    parallelized=True, rxType='xyz',
    Jpath=work_dir + out_dir + "SensitivityXYZ.zarr",
    equiSourceLayer=True)

survey.unpair()
survey.pair(prob)
xyz = prob.fields(mrec)
ndata = survey.nD

d_amp = np.sqrt(xyz[::3]**2. +
                xyz[1::3]**2. +
                xyz[2::3]**2.)

rxLoc = survey.srcField.rxList[0].locs

plt.figure()
ax1 = plt.subplot(2, 2, 1)
Utils.PlotUtils.plot2Ddata(rxLoc, xyz[::3], ax=ax1)
ax1 = plt.subplot(2, 2, 2)
Utils.PlotUtils.plot2Ddata(rxLoc, xyz[1::3], ax=ax1)
ax1 = plt.subplot(2, 2, 3)
Utils.PlotUtils.plot2Ddata(rxLoc, xyz[2::3], ax=ax1)
ax1 = plt.subplot(2, 2, 4)
Utils.PlotUtils.plot2Ddata(rxLoc, d_amp, ax=ax1)

# Write data out
Utils.io_utils.writeUBCmagneticsObservations(
    work_dir + out_dir + 'Bx.obs', survey, xyz[::3]
)
Utils.io_utils.writeUBCmagneticsObservations(
    work_dir + out_dir + 'By.obs', survey, xyz[1::3]
)
Utils.io_utils.writeUBCmagneticsObservations(
    work_dir + out_dir + 'Bz.obs', survey, xyz[2::3]
)
Utils.io_utils.writeUBCmagneticsObservations(
    work_dir + out_dir + 'Amplitude.obs', survey, d_amp
)
Utils.io_utils.writeUBCmagneticsObservations(
    work_dir + out_dir + 'Predicted.obs', survey, invProb.dpred
)
