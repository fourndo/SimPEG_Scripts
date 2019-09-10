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
#work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\\Research\\Synthetic\\Block_Gaussian_topo\\"
# work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Synthetic\\SingleBlock\\Simpeg\\"
#work_dir = "C:\\Egnyte\\Private\\dominiquef\\Projects\\4559_CuMtn_ZTEM\\Modeling\\MAG\\A1_Fenton\\"
work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Synthetic\\Nut_Cracker\\"
# work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\TKC\\DIGHEM_TMI\\"
#work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Synthetic\\Triple_Block_lined\\"
out_dir = "SimPEG_Octree_Susc_Inv\\"
input_file = "SimPEG_MAG.inp"
# %%
# Read in the input file which included all parameters at once
# (mesh, topo, model, survey, inv param, etc.)
driver = PF.MagneticsDriver.MagneticsDriver_Inv(work_dir + input_file)

os.system('if not exist ' + work_dir + out_dir + ' mkdir ' + work_dir+out_dir)

# Access the mesh and survey information
meshInput = driver.mesh
survey = driver.survey
actv = driver.activeCells

xyzLocs = survey.srcField.rxList[0].locs.copy()

if driver.topofile is not None:
    topo = np.genfromtxt(driver.basePath + driver.topofile,
                         skip_header=1)
    topo = topo[::100,:]
    xyzLocs = np.r_[xyzLocs, topo[::100, :]]

if isinstance(meshInput, Mesh.TensorMesh):
    # Define an octree mesh based on the provided tensor
    h = np.r_[meshInput.hx.min(), meshInput.hy.min(), meshInput.hz.min()]
    coreX, coreY, coreZ = meshInput.hx == h[0], meshInput.hy == h[1], meshInput.hz == h[2]
    padx, pady, padz = meshInput.hx[~coreX].sum(), meshInput.hy[~coreY].sum(), meshInput.hz[~coreZ].sum()

    padDist = np.r_[np.c_[padx, padx], np.c_[pady, pady], np.c_[padz, padz]]


    mesh = Utils.modelutils.meshBuilder(xyzLocs, h, padDist,
                                        meshGlobal=None,
                                        meshType='TREE',padCore=np.r_[3, 3, 3],
                                        gridLoc = 'CC')
else:
    mesh = Mesh.TreeMesh.readUBC(driver.basePath + driver.mshfile)

if driver.topofile is not None:
    # Find the active cells
    actv = Utils.surface2ind_topo(mesh, topo)
else:
    actv = mesh.gridCC[:, 2] <= meshInput.vectorNz[-1]

nC = int(np.sum(actv))

Mesh.TreeMesh.writeUBC(mesh, work_dir + out_dir + 'OctreeTest.msh',
                       models={work_dir + out_dir + 'ActiveOctree.dat': actv})

# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, 0)

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

wr /= mesh.vol[actv]
wr = (wr/np.max(wr))
wr = wr**0.5

Mesh.TreeMesh.writeUBC(mesh, work_dir + out_dir + 'OctreeTest.msh',
                       models={work_dir + out_dir + 'SensWeights.sus': actvMap*wr})

# wr = PF.Magnetics.get_dist_wgt(mesh, rxLoc, actv, 3, 1)

# Create a regularization
reg = Regularization.Tikhonov(mesh, indActive=actv, mapping=idenMap)
# reg.norms = driver.lpnorms
# # reg.alpha_s = 2.5e-3
# if driver.eps is not None:
#     reg.eps_p = driver.eps[0]
#     reg.eps_q = driver.eps[1]

reg.cell_weights = wr#driver.cell_weights*mesh.vol**0.5
reg.mref = np.zeros(nC)
# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=20, lower=0., upper=10.,
                                 maxIterLS=20, maxIterCG=10,
                                 tolCG=1e-4, tolG=1e-2)
invProb = InvProblem.BaseInvProblem(dmis, reg, opt, bfgs=False)
betaest = Directives.BetaEstimate_ByEig()

# Here is where the norms are applied
# Use pick a treshold parameter empirically based on the distribution of
#  model parameters
#IRLS = Directives.Update_IRLS(f_min_change=1e-3, minGNiter=3, maxIRLSiter=10)
update_Jacobi = Directives.UpdateJacobiPrecond()
BetaSchedule = Directives.BetaSchedule(coolingRate=1, coolingFactor=2.)
TargetMisfit = Directives.TargetMisfit()
#saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap)
#saveModel.fileName = work_dir + out_dir + 'ModelSus'

inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, BetaSchedule,TargetMisfit, update_Jacobi])

# Run the inversion
m0 = np.ones(nC)*1e-3  # Starting model
# prob.model = m0
mrec = inv.run(m0)

Mesh.TreeMesh.writeUBC(mesh, work_dir + out_dir + 'OctreeTest.msh',
                       models={work_dir + out_dir + 'Model_l2.sus': actvMap*invProb.l2model})
Mesh.TreeMesh.writeUBC(mesh, work_dir + out_dir + 'OctreeTest.msh',
                       models={work_dir + out_dir + 'Model_lp.sus': actvMap*mrec})
PF.Magnetics.writeUBCobs(work_dir + out_dir + 'Predicted_l2.pre',
                         survey, d=survey.dpred(invProb.l2model))

PF.Magnetics.writeUBCobs(work_dir + out_dir + 'Predicted_lp.pre',
                         survey, d=survey.dpred(invProb.model))
