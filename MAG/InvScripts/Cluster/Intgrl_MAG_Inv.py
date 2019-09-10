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
out_dir = "SimPEG_PF_Inv_MAG/"
tile_dirl2 = 'Tiles_l2/'
tile_dirlp = 'Tiles_lp/'
input_file = "SimPEG_MAG.inp"


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


os.system('mkdir ' + work_dir+out_dir+tile_dirl2)
os.system('mkdir ' + work_dir+out_dir+tile_dirlp)

nD = int(survey.nD*0.5)
print("nD ratio:" + str(nD) +'\\' + str(survey.nD) )
indx = np.random.randint(0, high=survey.nD, size=nD)
# Create a new downsampled survey
locXYZ = survey.srcField.rxList[0].locs[indx,:]

dobs = survey.dobs
std = survey.std

rxLoc = PF.BaseGrav.RxObs(locXYZ)
srcField = PF.BaseMag.SrcField([rxLoc], param=survey.srcField.param)
survey = PF.BaseMag.LinearSurvey(srcField)
survey.dobs = dobs[indx]
survey.std = std[indx]

PF.Magnetics.writeUBCobs(work_dir+out_dir + 'MAG_Inv_Sub.obs', survey, survey.dobs)



Mesh.TensorMesh.writeUBC(mesh, work_dir + out_dir + tile_dirl2 + "MAGsus" + ".msh")
Mesh.TensorMesh.writeUBC(mesh, work_dir + out_dir + tile_dirlp + "MAGsus" + ".msh")

nC = len(actv)
# print("Tile "+str(tt))
# Create active map to go from reduce space to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

# Create identity map
idenMap = Maps.IdentityMap(nP=nC)

mstart = np.ones(nC)*1e-4

# Create the forward model operator
prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap,
                                      actInd=actv)
prob.model = mstart

# Pair the survey and problem
survey.pair(prob)

wr = np.zeros(prob.F.shape[1])
for ii in range(survey.nD):
    wr += (prob.F[ii, :]/survey.std[ii])**2.

wr = (wr/np.max(wr))
wr = wr**0.5

# Create a sparse regularization
reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
reg.cell_weights = wr
reg.mref = np.zeros(nC)
reg.norms = driver.lpnorms
if driver.eps is not None:
    reg.eps_p = driver.eps[0]
    reg.eps_q = driver.eps[1]

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1/survey.std

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=50, lower=0., upper=1.,
                                 maxIterLS=20, maxIterCG=10,
                                 tolCG=1e-3)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

# Here is the list of directives
betaest = Directives.BetaEstimate_ByEig()

# Specify the sparse norms
IRLS = Directives.Update_IRLS(f_min_change=1e-3,
                              minGNiter=3, coolingRate=1, chifact=0.25,
                              maxIRLSiter=5)

# Special directive specific to the mag amplitude problem. The sensitivity
# weights are update between each iteration.
update_Jacobi = Directives.UpdatePreCond()

saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap)
saveModel.fileName = work_dir + out_dir + 'AmpInv'

# Put all together
inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, IRLS, update_Jacobi,
                                             saveModel])

# Invert
mrec = inv.run(mstart)

# Outputs
if getattr(invProb, 'l2model', None) is not None:
    Mesh.TensorMesh.writeModelUBC(mesh,work_dir + out_dir + tile_dirl2 + "MAGsus" + ".sus", actvMap*invProb.l2model)

Mesh.TensorMesh.writeModelUBC(mesh,work_dir + out_dir + tile_dirlp + "MAGsus" + ".sus", actvMap*invProb.model)
PF.Magnetics.writeUBCobs(work_dir+out_dir + 'MAG.pre', survey, invProb.dpred)


del prob
gc.collect()

#PF.Magnetics.plot_obs_2D(rxLoc,invProb.dpred,varstr='Amplitude Data')
