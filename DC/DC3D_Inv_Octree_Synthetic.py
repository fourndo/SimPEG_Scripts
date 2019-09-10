# -*- coding: utf-8 -*-
"""
Invert synthetic DC data with sensitivity weighting and compact norms

Created on Thu Jul 27 13:32:00 2017

@author: DominiqueFournier
"""

from SimPEG import Mesh, Utils, EM, Maps, Survey
from SimPEG import DataMisfit, Regularization, Optimization
from SimPEG import Directives, InvProblem, Inversion
from SimPEG.Utils import mkvc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from SimPEG.EM.Static import DC, IP
from pymatsolver import PardisoSolver


wrType = 'sensitivityW'
arrayType = 'dipole-dipole'
changeMref = False
mref_val = 1e-3
# Create topography
X, Y = np.meshgrid(np.linspace(-800,800,20), np.linspace(-800,800,20))
Z = np.ones_like(X)*0

# Create the mesh
csx, csy, csz = 50., 50., 50.
ncx, ncy, ncz, npad = 32, 16, 16, 10
hx = [(csx, npad, -1.3), (csx, ncx), (csx, npad, 1.3)]
hy = [(csy, npad, -1.3), (csy, ncy), (csy, npad, 1.3)]
# hz = [(csz,npad,-1.4), (csz,12), (50,6),(50,6),(csz,12), (csz,npad,1.4)]
hz = [(csz, npad, -1.3), (csz, 12), (50, 6)]
mesh = Mesh.TensorMesh([hx, hy, hz], 'CCN')
#mesh.x0 = mesh.x0 + np.r_[0,0,75]

srclocA = np.c_[-750., 0., -25.]
srclocB = np.c_[750., 0., -25.]
endl = np.r_[srclocA, srclocB]

survey = EM.Static.Utils.gen_DCIPsurvey(endl, mesh, arrayType, 100, 100, 10)

xyzLocs = survey.srcList[0].rxList[0].locs[0]
h = np.r_[mesh.hx.min(), mesh.hy.min(), mesh.hz.min()]
coreX, coreY, coreZ = mesh.hx == h[0], mesh.hy == h[1], mesh.hz == h[2]
padx, pady, padz = mesh.hx[~coreX].sum(), mesh.hy[~coreY].sum(), mesh.hz[~coreZ].sum()

padDist = np.r_[np.c_[padx, padx], np.c_[pady, pady], np.c_[padz, padz]]

mesh = Utils.modelutils.meshBuilder(np.r_[xyzLocs, np.c_[mkvc(X), mkvc(Y), mkvc(Z)]], h, padDist,
                                    padCore=np.r_[1, 1, 4], meshGlobal=None,
                                    meshType='TREE', gridLoc='N')


blkind1 = Utils.ModelBuilder.getIndicesBlock([-50., -500., -50.], [50., 500., -1050.], mesh.gridCC)
#blkind2 = Utils.ModelBuilder.getIndicesBlock([400., -250., -250.], [900., 250., -750.], mesh.gridCC)
airind = mesh.gridCC[:, 2] > -0.
overburden = mesh.gridCC[:, 2] > -200

background = 0.001
sigma = np.ones(mesh.nC)*background
sigma[blkind1] = 0.05
sigma[overburden] = 1e-3
sigma[airind] = 1e-8
# Create a flat topo
topoXYZ = Utils.ndgrid(mesh.vectorNx, mesh.vectorNy, np.r_[-1.])

actv = sigma != 1e-8

nC = int(actv.sum())

midLocs = survey.srcList[0].rxList[0].locs[0]+ survey.srcList[0].rxList[0].locs[1]
midLocs /= 2

idenMap = Maps.IdentityMap(nP=nC)
expmap = Maps.ExpMap(mesh)
actmap = Maps.InjectActiveCells(mesh, actv, np.log(1e-8))
mapping = expmap * actmap
logmap = Maps.LogMap(mesh)

m0 = np.log(np.ones_like(sigma)*1e-3)[actv]
mref = np.log(np.ones_like(sigma)*mref_val)

problem = DC.Problem3D_N(mesh, sigmaMap=mapping, storeJ=True)
problem.Solver = PardisoSolver
problem.pair(survey)
mtrue = np.log(sigma[actv])

# Depth weight
#depth = 1./(abs(mesh.gridCC[:,2]))**1.5
#depth = depth/depth.max()
d0 = survey.dpred(m0)
dobs = survey.makeSyntheticData(mtrue, std=0.02)

survey.std = abs(dobs) * 5e-2 + 1e-4

# EM.Static.Utils.writeUBC_DCobs('Survey_Synthetic.dat',survey,'3D','SURFACE')

#wd = 1./(abs(dobs)*0.005)
wd = 1./(survey.std)
# Create inversion objects and run
#TODO put warning when dobs is not set!
survey.dobs = dobs
dmisfit = DataMisfit.l2_DataMisfit(survey)
dmisfit.W = wd
reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
reg.norms = [2, 2, 2, 2]
reg.mref = mref[actv]
reg.eps_p = 1e-3
reg.eps_q = 1e-3

# Compute sensitivity weight

if wrType == 'depthW':
    wr = depth
    beta = 1e+2

elif wrType == 'noW':
    wr = depth**0.
    beta = 1e+2
elif wrType == 'sensitivityW':
    beta = 1e+2
    wr = np.sum((problem.getJ(m0, problem.fields(m0)))**2.,axis=0)**0.5
    wr = wr/wr.max()

#reg.cell_weights = wr
opt = Optimization.ProjectedGNCG(maxIter=20, lower=-10, upper=10,
                                  maxIterLS=20, maxIterCG=30, tolCG=1e-4)
#opt = Optimization.InexactGaussNewton(maxIter = 20)

invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt, beta=beta)
# Create an inversion object
beta = Directives.BetaSchedule(coolingFactor=2, coolingRate=2)
betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e0)
# save = Directives.SaveOutputEveryIteration()
target = Directives.TargetMisfit()
update_IRLS = Directives.Update_IRLS(f_min_change=1e-3, minGNiter=2, maxIRLSiter=10)
updateWr = Directives.UpdateSensWeighting()

inv = Inversion.BaseInversion(invProb, directiveList=[update_IRLS, updateWr])
problem.counter = opt.counter = Utils.Counter()
opt.LSshorten = 0.5
opt.remember('xc')
mopt = inv.run(m0)


#
#%% Plot the model
if changeMref:
    figName = 'DC_PDP_' + wrType + '_' + str(mref_val) + 'RefUpdate.png'

else:
    figName = 'DC_PDP_' + wrType + '_' + str(mref_val) + 'Ref.png'
mout = np.log10(mapping*mopt)

#mout = np.log10(sigma)
#figName = 'Mtrue.png'


vmin = -4
vmax = -2



indz= -5
indx1 = 19
indy1 = 18
xlim = [-1000, 1000]
ylim = [-1000, 1000]
zlim = [-1000, 25]

fig, ax = plt.figure(figsize = (6,8)), plt.subplot(2,1,1)
im = mesh.plotSlice(mout, ax=ax, grid=False, ind=indz, clim=(vmin, vmax))
plt.scatter(midLocs[:,0], midLocs[:,1], 2, color='k')
plt.scatter(srclocA[0,0], srclocA[0,1], 5, color='r')
plt.scatter(srclocB[0,0], srclocB[0,1], 5, color='r')

plt.plot(xlim, np.r_[mesh.vectorCCy[indy1],mesh.vectorCCy[indy1]], 'w--')
#plt.plot(np.r_[mesh.vectorCCx[indx1],mesh.vectorCCx[indx1]], ylim, 'w--')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
# ax.set_xlabel("Easting (m)")
ax.set_xlabel(" ")
ax.set_ylabel("Northing (m)")
ax.set_title(("Depth at %.1f m")%(mesh.vectorCCz[indz]))
#rectxyx=np.r_[-900., -400., -400., -900., -900.]
#rectxyy=np.r_[-250., -250., 250., 250., -250.]
#ax.plot(rectxyx, rectxyy,'k-',lw=3)
#ax.plot(-rectxyx, rectxyy,'k-',lw=3)
plt.colorbar(im[0])
ax.set_aspect('equal')

ax = plt.subplot(2,1,2)


im = mesh.plotSlice(mout, ax=ax, grid=False, normal='Y', clim=(vmin, vmax), ind = indy1)
plt.plot(xlim,np.r_[mesh.vectorCCz[indz],mesh.vectorCCz[indz]], 'w--')
ax.set_xlim(xlim)
ax.set_ylim(zlim)
ax.set_title(("Northing at %.1f m")%(mesh.vectorCCy[indy1]))
ax.set_xlabel("Easting (m)")
# ax.set_xlabel(" ")

ax.set_ylabel("Depth (m)")
#rectxzx=np.r_[-900., -400., -400., -900., -900.]
#rectxzz=np.r_[-700., -700., -250., -250., -700.]
#ax.plot(rectxzx, rectxzz,'k-',lw=3)
#ax.plot(-rectxzx, rectxzz,'k-',lw=3)
ax.set_aspect('equal')

#ax = plt.subplot(2,2,3)
#
#
#im = mesh.plotSlice(mout, ax=ax, grid=False, normal='X', clim=(vmin, vmax), ind = indx1)
#plt.plot(ylim,np.r_[mesh.vectorCCz[indz],mesh.vectorCCz[indz]], 'w--')
#ax.set_xlim(ylim)
#ax.set_ylim(zlim)
#ax.set_title(("Easting at %.1f m")%(mesh.vectorCCx[indx1]))
#ax.set_xlabel("Northing (m)")
## ax.set_xlabel(" ")
#
#ax.set_ylabel("Depth (m)")
#rectxzx=np.r_[-900., -400., -400., -900., -900.]
#rectxzz=np.r_[-700., -700., -250., -250., -700.]
#ax.plot(rectxzx, rectxzz,'k-',lw=3)
#ax.plot(-rectxzx, rectxzz,'k-',lw=3)
#ax.set_aspect('equal')
#
plt.savefig(figName, bbox_inches='tight')

#%% PLOT data
figName = 'DC_Obs_vs_Pred.png'
fig, ax1 = plt.figure(figsize=(8,8)), plt.subplot(2,1,1)

EM.Static.Utils.plot_pseudoSection(survey,ax1, surveyType =arrayType, scale='log')

ax2 = plt.subplot(2,1,2)
EM.Static.Utils.plot_pseudoSection(survey,ax2, dobs=invProb.dpred,surveyType =arrayType, scale='log')

fig.savefig(figName, bbox_inches='tight')
#%%
#import scipy.sparse as sp
#Japprox = np.sum((problem.getJ(mtrue, problem.fields(mtrue)))**2.,axis=0)

mesh.writeUBC('MeshDC.msh',{'InvModel.con':mapping*mopt})
mesh.writeUBC('MeshDC.msh',{'TrueModel.con':sigma})


#%%
#threshold = pctVal[2]
#step = mopt - m0
#
#indx = Japprox > threshold
#
#P = sp.spdiags(indx*1.,0, mesh.nC, mesh.nC)
#mtest = mopt.copy()
#mtest[indx==False] = np.log(1e-8)
#ddata = np.sum(((survey.dpred(mtest) - survey.dobs)*wd)**2.)/2
#print(ddata, threshold)
# plt.figure()
# plt.plot(np.dot(prob.G*P,step))
# plt.plot(survey.dobs)
