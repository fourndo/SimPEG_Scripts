# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:47:46 2017

@author: DominiqueFournier
"""

import numpy as np
import scipy.sparse as sp
import unittest
from SimPEG import Mesh, Maps, Models, Utils, PF, Regularization, Directives
from SimPEG import InvProblem, Optimization, Inversion, DataMisfit
import inspect
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from SimPEG.Utils import mkvc

lx = 1000
dxMin = 10

nC = 2**int(np.log2(lx/dxMin))
width = dxMin * nC
dxMax = width/4
maxLevel = int(np.log2(dxMax / dxMin))

# Define topo and refine around max values
topoLim = width/4
topoLim = np.linspace(-100, 100, 20)
X, Y = np.meshgrid(topoLim, topoLim)
Z = np.ones_like(X) * 0.5
topo = np.c_[mkvc(X), mkvc(Y), mkvc(Z)]

padDist = np.r_[np.c_[400, 400], np.c_[400, 400], np.c_[400, 400]]
mesh = Utils.modelutils.meshBuilder(topo, np.r_[dxMin, dxMin, dxMin], padDist,
                                    padCore=np.r_[0, 0, 3], meshGlobal=None,
                                    expFact=1.3,
                                    meshType='TREE')

plt.figure(figsize=(5, 5))
ax = plt.subplot(projection='3d')
mesh.plotGrid(ax=ax)


# Define active cells
zTopo = griddata(topo[:, :2], topo[:, 2],
                 mesh.gridCC[:, :2],
                 method='nearest',
                 fill_value=np.nan)
actInd = zTopo > mesh.gridCC[:, 2]

Mesh.TreeMesh.writeUBC(mesh, 'OctreeTest.msh',
                       models={'ActiveOctree.dat': actInd})

# Define the inducing field parameter
B = (50000, 90, 0)

# Create and array of observation points
xr = np.linspace(-30., 30., 20)
yr = np.linspace(-30., 30., 20)
X, Y = np.meshgrid(xr, yr)

# Move the observation points 5m above the topo
Z = np.ones(X.shape)*topo[:, 2].max() + 10

# Create a MAGsurvey
xyzLoc = np.c_[Utils.mkvc(X.T), Utils.mkvc(Y.T), Utils.mkvc(Z.T)]
rxLoc = PF.BaseMag.RxObs(xyzLoc)
srcField = PF.BaseMag.SrcField([rxLoc], param=B)
survey = PF.BaseMag.LinearSurvey(srcField)

# We can now create a susceptibility model and generate data
# Here a simple block in half-space
susc = 0.1
nX = 1

# Get index of the center of block
midx = 0
midy = 0
midz = -30

model = np.zeros(mesh.nC)
p0 = np.r_[midx, midy, midz] - nX*dxMin
p1 = np.r_[midx, midy, midz] + nX*dxMin
actBlock = Utils.ModelBuilder.getIndicesBlock(p0, p1, mesh.gridCC)

model[actBlock] = susc
m = model[actInd]

# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actInd, -100)

Mesh.TreeMesh.writeUBC(mesh, 'OctreeTest.msh',
                       models={'OctreeModel.dat': actvMap*m})
# G=mesh._cellGradyStencil()
# gradm = G.T*(G*model)
# mesh.cellGradStencil.shape

# Number of active cells
nC = int(actInd.sum())

# Create reduced identity map
idenMap = Maps.IdentityMap(nP=nC)

# Create the forward problem (forwardOnly)
prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap, actInd=actInd)

# Pair the survey and problem
survey.pair(prob)

# Compute forward model some data
d = prob.fields(m)

nD = survey.nD

wd = np.ones(len(d))  # Assign flat uncertainties
survey.dobs = d
survey.std = wd

PF.Magnetics.writeUBCobs('SimulatedData.obs', survey, d)
fig, im = PF.Magnetics.plot_obs_2D(survey.srcField.rxList[0].locs, d=d)
plt.show()

# Create a regularization function, in this case l2l2
wr = np.sum(prob.G**2., axis=0)**0.5
wr = (wr/np.max(wr))

# Create a regularization
reg_Susc = Regularization.Sparse(mesh, indActive=actInd, mapping=idenMap)
reg_Susc.norms = [0, 1, 1, 1]
reg_Susc.cell_weights = wr

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1/wd

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=100, lower=0., upper=1.,
                                 maxIterLS=20, maxIterCG=10, tolCG=1e-3)
invProb = InvProblem.BaseInvProblem(dmis, reg_Susc, opt)
betaest = Directives.BetaEstimate_ByEig()

# Here is where the norms are applied
# Use pick a treshold parameter empirically based on the distribution of
#  model parameters
IRLS = Directives.Update_IRLS(f_min_change=1e-3, minGNiter=3)

update_Jacobi = Directives.UpdateJacobiPrecond()
inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest,
                                             IRLS,
                                             update_Jacobi])

# Run the inversion
m0 = np.ones(nC)*1e-4  # Starting model
mrec_SUS = inv.run(m0)

Mesh.TreeMesh.writeUBC(mesh, 'OctreeTest.msh',
                       models={'Model_l2.sus': actvMap*IRLS.l2model})
Mesh.TreeMesh.writeUBC(mesh, 'OctreeTest.msh',
                       models={'Model_lp.sus': actvMap*mrec_SUS})
