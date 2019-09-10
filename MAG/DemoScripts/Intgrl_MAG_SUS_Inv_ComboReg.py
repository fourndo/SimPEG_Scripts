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

#work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Kevitsa\\Modeling\\MAG\\"
#work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\\Research\\Modelling\\Synthetic\\Block_Gaussian_topo\\"
work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Synthetic\\SingleBlock\\Simpeg\\"
#work_dir = "C:\\Egnyte\\Private\\dominiquef\\Projects\\4559_CuMtn_ZTEM\\Modeling\\MAG\\A1_Fenton\\"
# work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Modelling\\Synthetic\\Nut_Cracker\\"
# work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\TKC\\DIGHEM_TMI\\"
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


nC = len(actv)


# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

# Creat reduced identity map
idenMap = Maps.IdentityMap(nP=nC)

# Create the forward model operator
prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap, actInd=actv)

# Pair the survey and problem
survey.pair(prob)

# Create sensitivity weights from our linear forward operator
rxLoc = survey.srcField.rxList[0].locs
wr = np.zeros(prob.F.shape[1])
for ii in range(survey.nD):
    wr += (prob.F[ii, :]/survey.std[ii])**2.
wr = wr**0.5
wr = (wr/np.max(wr))


# Create a regularization
reg1 = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
reg1.norms = driver.lpnorms
reg1.alpha_s = reg1.alpha_x = reg1.alpha_y = reg1.alpha_z = 1e-1
if driver.eps is not None:
    reg1.eps_p = driver.eps[0]
    reg1.eps_q = driver.eps[1]

# reg1.cell_weights = mesh.vol#np.ones(nC)#driver.cell_weights*
reg1.mref = driver.mref

# Create a regularization for sensitivity weighting
reg2 = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
#reg2.alpha_x, reg2.alpha_y, reg2.alpha_z = 0, 0 ,0

reg2.cell_weights = wr#*10*mesh.vol.min()#driver.cell_weights*mesh.vol**0.5
reg2.mref = driver.mref

reg= reg1 + reg2
# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=20, lower=0., upper=10.,
                                 maxIterLS=20, maxIterCG=10, tolCG=1e-4)
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
betaest = Directives.BetaEstimate_ByEig()

# Here is where the norms are applied
# Use pick a treshold parameter empirically based on the distribution of
#  model parameters
IRLS = Directives.Update_IRLS(f_min_change=1e-3, minGNiter=3, maxIRLSiter=10)
update_Jacobi = Directives.UpdatePreCond()
#ScaleComboReg = Directives.ScaleComboReg()
saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap)
saveModel.fileName = work_dir + out_dir + 'ModelSus_1e1'

inv = Inversion.BaseInversion(invProb,
                              directiveList=[ betaest, IRLS,  update_Jacobi,  saveModel])

# Run the inversion
m0 = driver.m0  # Starting model
# prob.model = m0
mrec = inv.run(m0)

#%%
plt.figure()
axs = plt.subplot()
indx = np.argsort(invProb.model)
reg.objfcts[0].objfcts[0]._stashedR=None
reg.model = invProb.model
axs.plot(invProb.model[indx],reg.objfcts[1].objfcts[0].deriv(invProb.model)[indx])
axs.plot(invProb.model[indx],reg.objfcts[0].objfcts[0].deriv(invProb.model)[indx])
axs.plot(invProb.model[indx],reg.objfcts[0].objfcts[1].deriv(invProb.model)[indx])
