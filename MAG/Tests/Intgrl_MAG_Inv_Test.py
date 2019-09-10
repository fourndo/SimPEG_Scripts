# -*- coding: utf-8 -*-
"""
Run a simple cube inversion

Created on Thu Sep 29 10:11:11 2016

@author: dominiquef
"""
from SimPEG import *
import SimPEG.PF as PF
#import simpegCoordUtils as Utils
import matplotlib.pyplot as plt

# Define the inducing field parameter
H0 = (50000, 90, 0)

# Create a mesh
dx = 5.

hxind = [(dx, 5, -1.3), (dx, 9), (dx, 5, 1.3)]
hyind = [(dx, 5, -1.3), (dx, 9), (dx, 5, 1.3)]
hzind = [(dx, 5, -1.3), (dx, 6)]

mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCC')

# Get index of the center
midx = int(mesh.nCx/2)
midy = int(mesh.nCy/2)

# Lets create a simple Gaussian topo and set the active cells
[xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
zz = np.ones(xx.shape)*mesh.vectorNz[-1]

# Go from topo to actv cells
topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]
actv = Utils.surface2ind_topo(mesh, topo, 'N')
actv = np.asarray([inds for inds, elem in enumerate(actv, 1)
                  if elem], dtype=int) - 1

# Create active map to go from reduce space to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)
nC = len(actv)

# Create and array of observation points
xr = np.linspace(-20., 20., 20)
yr = np.linspace(-20., 20., 20)
X, Y = np.meshgrid(xr, yr)

# Move the observation points 5m above the topo
Z = np.ones(X.shape)*mesh.vectorNz[-1] + dx

# Create a MAGsurvey
rxLoc = np.c_[Utils.mkvc(X.T), Utils.mkvc(Y.T), Utils.mkvc(Z.T)]
rxLoc = PF.BaseMag.RxObs(rxLoc)
srcField = PF.BaseMag.SrcField([rxLoc], param=H0)
survey = PF.BaseMag.LinearSurvey(srcField)

# We can now create a susceptibility model and generate data
# Here a simple block in half-space
model = np.zeros((mesh.nCx, mesh.nCy, mesh.nCz))
model[(midx-1):(midx+2), (midy-1):(midy+2), -6:-3] = 0.02
model = Utils.mkvc(model)
model = model[actv]

# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

# Creat reduced identity map
idenMap = Maps.IdentityMap(nP=nC)

# Create the forward model operator
prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap,
                                     actInd=actv)

# Pair the survey and problem
survey.pair(prob)

# Compute linear forward operator and compute some data
d = prob.fields(model)

# Add noise and uncertainties (1nT)
data = d + np.random.randn(len(d))
wd = np.ones(len(data))*1.

survey.dobs = data
survey.std = wd

# Create sensitivity weights from our linear forward operator
wr = np.sum(prob.G**2., axis=0)**0.5
wr = (wr/np.max(wr))

# Create a regularization
reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
reg.cell_weights = wr
reg.mref = np.zeros(nC)

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = 1./wd

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=100, lower=0., upper=1.,
                                 maxIterLS=20, maxIterCG=10,
                                 tolCG=1e-4)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
betaest = Directives.BetaEstimate_ByEig()

# Here is where the norms are applied
IRLS = Directives.Update_IRLS(norms=([0, 1, 1, 1]),
                              eps=None, f_min_change=1e-3,
                              minGNiter=3)
update_Jacobi = Directives.Update_lin_PreCond(mapping=prob.chiMap)
inv = Inversion.BaseInversion(invProb,
                                   directiveList=[IRLS, betaest,
                                                  update_Jacobi])
                                                  
mrec = inv.run(np.ones(len(actv))*1e-4)

ypanel = midx
zpanel = -4
m_l2 = actvMap * reg.l2model
m_l2[m_l2==-100] = np.nan

m_lp = actvMap * mrec
m_lp[m_lp==-100] = np.nan
             
m_true = actvMap * model
m_true[m_true==-100] = np.nan
#%% Plot the result
fig = plt.figure()
ax2 = plt.subplot(221)
ax3 = plt.subplot(222)
ax4 = plt.subplot(223)
ax5 = plt.subplot(224)
                            
im2 = mesh.plotSlice(m_lp, ax = ax2, normal = 'Z', ind=zpanel, grid=True, clim = (0., m_l2.max()),pcolorOpts={'cmap':'viridis'})
#plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCy[ypanel],mesh.vectorCCy[ypanel]]),color='w')
plt.title('Plan l2-model.')
ax2.set_aspect('equal')
plt.ylabel('y')
ax2.xaxis.set_visible(False)
ax2.set_xlim(-60,60)
ax2.set_ylim(-60,60)    
ax2.axis('equal')

im3 = mesh.plotSlice(m_lp, ax = ax3, normal = 'Y', ind=midx, grid=True, clim = (0., m_l2.max()),pcolorOpts={'cmap':'viridis'})
#plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCz[zpanel],mesh.vectorCCz[zpanel]]),color='w')
#plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([Z.min(),Z.max()]),color='k')
plt.title('E-W l2-model.')

plt.xlabel('x')
plt.ylabel('z')
ax3.set_xlim(-60,60)
ax3.set_ylim(-80,0)  
ax3.set_aspect('equal') 
ax3.axis('equal')                                                

im2 = mesh.plotSlice(m_true, ax = ax4, normal = 'Z', ind=zpanel, grid=True, clim = (0., m_l2.max()),pcolorOpts={'cmap':'viridis'})
#plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCy[ypanel],mesh.vectorCCy[ypanel]]),color='w')
plt.title('Plan l2-model.')
ax4.set_aspect('equal')
plt.ylabel('y')
ax4.xaxis.set_visible(False)
ax4.set_xlim(-60,60)
ax4.set_ylim(-60,60)    
ax4.axis('equal')

im3 = mesh.plotSlice(m_true, ax = ax5, normal = 'Y', ind=midx, grid=True, clim = (0., m_l2.max()),pcolorOpts={'cmap':'viridis'})
#plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCz[zpanel],mesh.vectorCCz[zpanel]]),color='w')
#plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([Z.min(),Z.max()]),color='k')
plt.title('E-W l2-model.')

plt.xlabel('x')
plt.ylabel('z')
ax5.set_xlim(-60,60)
ax5.set_ylim(-80,0)  
ax5.set_aspect('equal') 
ax5.axis('equal')    