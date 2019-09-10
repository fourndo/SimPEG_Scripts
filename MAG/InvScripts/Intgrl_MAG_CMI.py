"""
This script runs a Cooperative Magnetic Inversion. The code combines
the Amplitude inversion and MVI-Cartesian formulation. The Amplitude inversion
must be runned ahead of time and inputed as a starting model in the input file.
The goal is to improve the MVI result by constraining the location and shape
of magnetic bodies.

-------------------------------------------------------------------------------
 !! IMPORTANT: PLease run Intgrl_MAG_AMP_Inv.py before running this script !!
-------------------------------------------------------------------------------


Created on Thu Sep 29 10:11:11 2016

@author: dominiquef
"""
from SimPEG import Mesh, Directives, Maps, InvProblem, Optimization
from SimPEG import DataMisfit, Inversion, Regularization
import SimPEG.PF as PF
import numpy as np
import os

# Define the inducing field parameter
work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\\Research\\Modelling\\Synthetic\\Block_Gaussian_topo\\"
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

# Extract active region
actv = driver.activeCells


# Use the amplitude model to prime the inversion starting model
m_amp = driver.m0

# Create rescaled weigths
mamp = (m_amp/m_amp.max() + 1e-2)**-1.

# Create active map to go from reduce space to full
actvMap = Maps.InjectActiveCells(mesh, actv, 0)
nC = int(len(actv))

# Create identity map
# Create wires to link the regularization to each model blocks
wires = Maps.Wires(('prim', nC),
           ('second', nC),
           ('third', nC))

# Create identity map
idenMap = Maps.IdentityMap(nP=3*nC)

# Create the forward model operator
prob = PF.Magnetics.MagneticVector(mesh, chiMap=idenMap,
                                     actInd=actv)

# Pair the survey and problem
survey.pair(prob)


# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std



wr = np.sum(prob.F**2., axis=0)**0.5
wr = (wr/np.max(wr))*np.r_[mamp, mamp, mamp]


# Create a regularization
reg_p = Regularization.Sparse(mesh, indActive=actv, mapping=wires.prim)
reg_p.cell_weights = wires.prim * wr

reg_s = Regularization.Sparse(mesh, indActive=actv, mapping=wires.second)
reg_s.cell_weights = wires.second * wr

reg_t = Regularization.Sparse(mesh, indActive=actv, mapping=wires.third)
reg_t.cell_weights = wires.third * wr

reg = reg_p + reg_s + reg_t


# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=30, lower=-10., upper=10.,
                                 maxIterCG=20, tolCG=1e-3)


invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
betaest = Directives.BetaEstimate_ByEig()

betaCool = Directives.BetaSchedule(coolingFactor=2., coolingRate=1)

update_Jacobi = Directives.UpdatePreCond()

targetMisfit = Directives.TargetMisfit()

saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap)
saveModel.fileName = work_dir + out_dir + 'CMI_pst'
inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, update_Jacobi, betaCool,
                                             targetMisfit, saveModel])

mstart= np.ones(3*len(actv))*1e-4
mrec = inv.run(mstart)


PF.Magnetics.writeUBCobs(work_dir+out_dir + 'CMI.pre', survey, invProb.dpred)
