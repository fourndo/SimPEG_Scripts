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
from SimPEG.Utils import mkvc
import SimPEG.PF as PF
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import gc

#work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Kevitsa\\Modeling\\MAG\\"
#work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\\Research\\Modelling\\Synthetic\\Block_Gaussian_topo\\"
# work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Modelling\\Synthetic\\Nut_Cracker\\"
#work_dir = "C:\\Egnyte\\Private\\dominiquef\\Projects\\4559_CuMtn_ZTEM\\Modeling\\MAG\\A1_Fenton\\"
#work_dir = 'C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Kevitsa\\Modeling\\MAG\\Aiborne\\'
work_dir = '/tera_raid/dfournier/Kevitsa/MAG/Aiborne/'
out_dir = "SimPEG_PF_Inv_MAI/"
tile_dirl2 = 'Tiles_l2/'
tile_dirlp = 'Tiles_lp/'
input_file = "SimPEG_MAI.inp"


# %%
# Read in the input file which included all parameters at once
# (mesh, topo, model, survey, inv param, etc.)
driver = PF.MagneticsDriver.MagneticsDriver_Inv(work_dir + input_file)

os.system('mkdir ' + work_dir+out_dir)
os.system('mkdir ' + work_dir+out_dir + tile_dirlp)
os.system('mkdir ' + work_dir+out_dir + tile_dirl2)

# Access the mesh and survey information
mesh = driver.mesh
survey = driver.survey
actv = driver.activeCells

os.system('if not exist ' + work_dir + out_dir + tile_dirl2 + ' mkdir ' + work_dir+out_dir+tile_dirl2)
os.system('if not exist ' + work_dir + out_dir + tile_dirlp + ' mkdir ' + work_dir+out_dir+tile_dirlp)


Mesh.TensorMesh.writeUBC(mesh, work_dir + out_dir + tile_dirl2 + "MAI" + ".msh")
Mesh.TensorMesh.writeUBC(mesh, work_dir + out_dir + tile_dirlp + "MAI" + ".msh")

nC = len(actv)

# Create active map to go from reduce space to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

# Create identity map
idenMap = Maps.IdentityMap(nP=nC)

mstart = np.ones(nC)*1e-4

# Create the forward model operator
prob = PF.Magnetics.MagneticAmplitude(mesh, chiMap=idenMap,
                                      actInd=actv)
prob.model = mstart

# Pair the survey and problem
survey.pair(prob)


# Create a sparse regularization
reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
reg.mref = np.zeros(nC)
reg.norms = driver.lpnorms
if driver.eps is not None:
    reg.eps_p = driver.eps[0]
    reg.eps_q = driver.eps[1]

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1/survey.std

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=100, lower=0., upper=1.,
                                 maxIterLS=20, maxIterCG=10,
                                 tolCG=1e-3)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

# Here is the list of directives
betaest = Directives.BetaEstimate_ByEig()

# Specify the sparse norms
IRLS = Directives.Update_IRLS(f_min_change=1e-3,
                              minGNiter=3, coolingRate=1, chifact=0.25,
                              maxIRLSiter=10)

# Special directive specific to the mag amplitude problem. The sensitivity
# weights are update between each iteration.
update_SensWeight = Directives.UpdateSensWeighting()
update_Jacobi = Directives.UpdateJacobiPrecond()

saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap)
saveModel.fileName = work_dir + out_dir + 'AmpInv'

# Put all together
inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, IRLS, update_SensWeight, update_Jacobi,
                                             saveModel])

# Invert
mrec = inv.run(mstart)

# Outputs
if getattr(invProb, 'l2model', None) is not None:
    Mesh.TensorMesh.writeModelUBC(mesh,work_dir + out_dir + tile_dirl2 + "MAI"  + ".sus", actvMap*invProb.l2model)
Mesh.TensorMesh.writeModelUBC(mesh,work_dir + out_dir + tile_dirlp + "MAI"  + ".sus", actvMap*invProb.model)
PF.Magnetics.writeUBCobs(work_dir+out_dir + 'MAI.pre', survey, invProb.dpred)


del prob
gc.collect()

#PF.Magnetics.plot_obs_2D(rxLoc,invProb.dpred,varstr='Amplitude Data')
