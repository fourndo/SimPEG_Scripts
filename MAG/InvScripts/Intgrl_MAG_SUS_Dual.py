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
from SimPEG import Mesh, Directives, Maps, InvProblem, Optimization, DataMisfit, Inversion, Utils, Regularization
import SimPEG.PF as PF
import numpy as np
import matplotlib.pyplot as plt
import os
from pymatsolver import PardisoSolver

#work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Kevitsa\\Modeling\\MAG\\"
work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\\Research\\Synthetic\\Block_Gaussian_topo\\"
# work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Synthetic\\SingleBlock\\Simpeg\\"
#work_dir = "C:\\Egnyte\\Private\\dominiquef\\Projects\\4559_CuMtn_ZTEM\\Modeling\\MAG\\A1_Fenton\\"
#work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Synthetic\\Nut_Cracker\\"
# work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\TKC\\DIGHEM_TMI\\"
out_dir = "SimPEG_Susc_Dual_Inv\\"
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


nC = len(actv)


# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

wires = Maps.Wires(('m1', nC), ('m2', nC))

# Create Sum map
sumMap = Maps.Sum([wires.m1, wires.m2])

# Create the forward model operator
prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=sumMap, actInd=actv)

# Pair the survey and problem
survey.pair(prob)

# Create sensitivity weights from our linear forward operator
rxLoc = survey.srcField.rxList[0].locs
wr = np.zeros(prob.F.shape[1])
for ii in range(survey.nD):
    wr += (prob.F[ii, :]/survey.std[ii])**2.

wr = (wr/np.max(wr))
wr = wr**0.5

Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + 'SensWeights.sus',
                              actvMap*wr)

#%% Create a regularization
reg_m1 = Regularization.Sparse(mesh, indActive=actv, mapping=wires.m1)
reg_m1.cell_weights = wr
if driver.eps is not None:
    reg_m1.eps_p = driver.eps[0]
    reg_m1.eps_q = driver.eps[1]
reg_m1.norms = [2, 2, 2, 2]
reg_m1.mref = wires.m1.P.T * driver.mref


reg_m2 = Regularization.Sparse(mesh, indActive=actv, mapping=wires.m2)
reg_m2.cell_weights = wr
#reg_m2.alpha_s = 0
if driver.eps is not None:
    reg_m2.eps_p = driver.eps[0]
    reg_m2.eps_q = driver.eps[1]
reg_m2.norms = [1, 0, 0, 0]
reg_m2.mref = wires.m2.P.T * driver.mref

reg = reg_m1 + reg_m2

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=20, lower=0., upper=10.,
                                 maxIterLS=20, maxIterCG=20, tolCG=1e-4)
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
betaest = Directives.BetaEstimate_ByEig()

# Here is where the norms are applied
# Use pick a treshold parameter empirically based on the distribution of
#  model parameters
IRLS = Directives.Update_IRLS(f_min_change=1e-3, minGNiter=2, maxIRLSiter=10,
                              chifact_start=100)
update_Jacobi = Directives.UpdateJacobiPrecond()

# saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap)
# saveModel.fileName = work_dir + out_dir + 'ModelSus'

inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, IRLS, update_Jacobi])

# Run the inversion
m0 = np.kron(np.r_[1,1], driver.m0)  # Starting model
# prob.model = m0
mrec = inv.run(m0)

#Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + 'ModelSus_m1_l2.sus',
#                              actvMap*(wires.m1*invProb.l2model))

Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + 'ModelSus_m1_lp.sus',
                              actvMap*(wires.m1*invProb.model))

#Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + 'ModelSus_m2_l2.sus',
#                              actvMap*(wires.m2*invProb.l2model))

Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + 'ModelSus_m2_lp.sus',
                              actvMap*(wires.m2*invProb.model))

#Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + 'ModelSus_tot_l2.sus',
#                              actvMap*(sumMap*invProb.l2model))

Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + 'ModelSus_tot_lp.sus',
                              actvMap*(sumMap*invProb.model))

#PF.Magnetics.writeUBCobs(work_dir + out_dir + 'Predicted_m1.pre',
#                         survey, d=survey.dpred(wires.m1*invProb.model))
#
#PF.Magnetics.writeUBCobs(work_dir + out_dir + 'Predicted_m2.pre',
#                         survey, d=survey.dpred(wires.m2*invProb.model))

PF.Magnetics.writeUBCobs(work_dir + out_dir + 'Predicted_full.pre',
                         survey, d=survey.dpred(invProb.model))