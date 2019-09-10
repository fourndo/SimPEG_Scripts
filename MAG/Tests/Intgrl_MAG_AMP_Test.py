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
H0 = (50000, 45, 0)

# Create a mesh
dx = 5.

hxind = [(dx, 5, -1.3), (dx, 20), (dx, 5, 1.3)]
hyind = [(dx, 5, -1.3), (dx, 20), (dx, 5, 1.3)]
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
#actv = np.asarray([inds for inds, elem in enumerate(actv, 1)
#                  if elem], dtype=int) - 1

# Create active map to go from reduce space to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)
nC = len(actv)

# Create and array of observation points
xr = np.linspace(-40., 40., 10)
yr = np.linspace(-40., 40., 10)
X, Y = np.meshgrid(xr, yr)

# Move the observation points 5m above the topo
Z = np.ones(X.shape)*mesh.vectorNz[-1] + dx

# Create a MAGsurvey
rxLoc = np.c_[Utils.mkvc(X.T), Utils.mkvc(Y.T), Utils.mkvc(Z.T)]
rxLoc = PF.BaseMag.RxObs(rxLoc)
srcField = PF.BaseMag.SrcField([rxLoc], param=H0)
survey = PF.BaseMag.LinearSurvey(srcField, rxType = 'xyz')

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

# Generate TMI data
# Create the forward model operator
prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap, actInd=actv)

# Pair the survey and problem
survey.pair(prob)

# Compute linear forward operator and compute some data
d_TMI = prob.fields(model)

survey.dobs = d_TMI
survey.std = np.ones_like(d_TMI)

# %% STEP 1: EQUIVALENT SOURCE LAYER
# Now that we have a model and a survey we can build the linear system ...
# Create the forward model operator

# Get the active cells for equivalent source is the top only
surf = Utils.surface2ind_topo(mesh, topo, 'N', layer=True)

nC = np.sum(surf)  # Number of active cells

# Create active map to go from reduce set to full
surfMap = Maps.InjectActiveCells(mesh, surf, -100)

# Create identity map
idenMap = Maps.IdentityMap(nP=nC)

# Create static map
prob = PF.Magnetics.MagneticIntegral(mesh, chiMap = idenMap, actInd=surf, equiSourceLayer = True)

prob.solverOpts['accuracyTol'] = 1e-4

# Pair the survey and problem
survey.pair(prob)

reg = Regularization.Simple(mesh, indActive=surf)
reg.mref = np.zeros(nC)

# Specify how the optimization will proceed
opt = Optimization.ProjectedGNCG(maxIter=150, lower=-np.inf,
                                 upper=np.inf, maxIterLS=20,
                                 maxIterCG=20, tolCG=1e-3)

# Define misfit function (obs-calc)
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = 1./survey.std

# create the default L2 inverse problem from the above objects
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

# Specify how the initial beta is found
betaest = Directives.BetaEstimate_ByEig()

# Beta schedule for inversion
betaSchedule = Directives.BetaSchedule(coolingFactor = 2., coolingRate = 1)

# Target misfit to stop the inversion
targetMisfit = Directives.TargetMisfit(chifact=0.1)

# Create combined the L2 and Lp problem
inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, betaSchedule, targetMisfit])

# Run L2 and Lp inversion
mstart = np.zeros(nC)
mrec = inv.run(mstart)


#Mesh.TensorMesh.writeModelUBC(mesh,work_dir + "EquivalentSource.sus",surfMap*mrec)


#%% STEP 2: COMPUTE AMPLITUDE DATA

prob.forwardOnly=True
prob.rtype='xyz'
pred = prob.Intrgl_Fwr_Op(m=mrec)

ndata = survey.nD

damp = np.sqrt(pred[:ndata]**2. +
                       pred[ndata:2*ndata]**2. +
                       pred[2*ndata:]**2.)

rxLoc = survey.srcField.rxList[0].locs

#%% STEP 3: RUN AMPLITUDE INVERSION

# Create active map to go from reduce space to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)
nC = len(actv)

# Create identity map
idenMap = Maps.IdentityMap(nP=nC)

mstart= np.ones(len(actv))*1e-4

# Create the forward model operator
survey.unpair()

prob = PF.Magnetics.MagneticAmplitude(mesh, chiMap=idenMap,
                                     actInd=actv)
                                     


prob.chi = mstart

# Change the survey to xyz components
survey.srcField.rxList[0].rxType = 'xyz'

# Pair the survey and problem
survey.pair(prob)

Bamp_true = prob.fields(model)
# Compute linear forward operator and compute some data

#PF.Magnetics.plot_obs_2D(survey.srcField.rxList[0].locs,d)

# Add noise and uncertainties (1nT)

survey.dobs = damp

# Create sensitivity weights from our linear forward operator
#wr = np.sum(prob.G**2., axis=0)**0.5
#wr = (wr/np.max(wr))

# Create a regularization
reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
reg.mref = np.zeros(nC)

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = 1/survey.std

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=100, lower=0., upper=1.,
                                 maxIterLS=20, maxIterCG=10,
                                 tolCG=1e-3)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta= 1e-3)
betaest = Directives.BetaEstimate_ByEig()

# Here is where the norms are applied
IRLS = Directives.Update_IRLS(norms=([0, 1, 1, 1]),
                              eps=None, f_min_change=1e-3,
                              minGNiter=5, coolingRate=5)
update_Jacobi = Directives.Amplitude_Inv_Iter()
inv = Inversion.BaseInversion(invProb,
                                   directiveList=[update_Jacobi,IRLS,betaest])

mrec = inv.run(mstart)

ypanel = midx
zpanel = -4
m_l2 = actvMap * reg.l2model
m_l2[m_l2==-100] = np.nan

m_lp = actvMap * mrec
m_lp[m_lp==-100] = np.nan
               
#%% Plot the result
PF.Magnetics.plot_obs_2D(rxLoc,d_TMI,varstr='TMI Data')
PF.Magnetics.plot_obs_2D(rxLoc,damp,varstr='EQS Amplitude Data')
PF.Magnetics.plot_obs_2D(rxLoc,Bamp_true,varstr='True Amplitude Data')

fig = plt.figure()
ax2 = plt.subplot(231)
ax3 = plt.subplot(234)
                                 
im2 = mesh.plotSlice(m_l2, ax = ax2, normal = 'Z', ind=zpanel, grid=True, clim = (0., m_l2.max()),pcolorOpts={'cmap':'viridis'})
#plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCy[ypanel],mesh.vectorCCy[ypanel]]),color='w')
plt.title('Plan l2-model.')
ax2.set_aspect('equal')
plt.ylabel('y')
ax2.xaxis.set_visible(False)
ax2.set_xlim(-60,60)
ax2.set_ylim(-60,60)    
ax2.axis('equal')

im3 = mesh.plotSlice(m_l2, ax = ax3, normal = 'Y', ind=midx, grid=True, clim = (0., m_l2.max()),pcolorOpts={'cmap':'viridis'})
#plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCz[zpanel],mesh.vectorCCz[zpanel]]),color='w')
#plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([Z.min(),Z.max()]),color='k')
plt.title('E-W l2-model.')

plt.xlabel('x')
plt.ylabel('z')
ax3.set_xlim(-60,60)
ax3.set_ylim(-80,0)  
ax3.set_aspect('equal') 
ax3.axis('equal')                                                
plt.colorbar(im3[0],ax=ax3, orientation='horizontal')

ax2 = plt.subplot(232)
ax3 = plt.subplot(235)
                                 
im2 = mesh.plotSlice(m_lp, ax = ax2, normal = 'Z', ind=zpanel, grid=True, clim = (0., m_lp.max()),pcolorOpts={'cmap':'viridis'})
#plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCy[ypanel],mesh.vectorCCy[ypanel]]),color='w')
plt.title('Plan l2-model.')
ax2.set_aspect('equal')
plt.ylabel('y')
ax2.xaxis.set_visible(False)
ax2.set_xlim(-60,60)
ax2.set_ylim(-60,60)    
ax2.axis('equal')

im3 = mesh.plotSlice(m_lp, ax = ax3, normal = 'Y', ind=midx, grid=True, clim = (0., m_lp.max()),pcolorOpts={'cmap':'viridis'})
#plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCz[zpanel],mesh.vectorCCz[zpanel]]),color='w')
#plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([Z.min(),Z.max()]),color='k')
plt.title('E-W l2-model.')

plt.xlabel('x')
plt.ylabel('z')
ax3.set_xlim(-60,60)
ax3.set_ylim(-80,0)  
ax3.set_aspect('equal') 
ax3.axis('equal')                                                
plt.colorbar(im3[0],ax=ax3, orientation='horizontal')

ax2 = plt.subplot(233)
ax3 = plt.subplot(236)
                                 
im2 = mesh.plotSlice(model, ax = ax2, normal = 'Z', ind=zpanel, grid=True, clim = (0., m_lp.max()),pcolorOpts={'cmap':'viridis'})
#plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCy[ypanel],mesh.vectorCCy[ypanel]]),color='w')
plt.title('Plan l2-model.')
ax2.set_aspect('equal')
plt.ylabel('y')
ax2.xaxis.set_visible(False)
ax2.set_xlim(-60,60)
ax2.set_ylim(-60,60)    
ax2.axis('equal')

im3 = mesh.plotSlice(model, ax = ax3, normal = 'Y', ind=midx, grid=True, clim = (0., m_lp.max()),pcolorOpts={'cmap':'viridis'})
#plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCz[zpanel],mesh.vectorCCz[zpanel]]),color='w')
#plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([Z.min(),Z.max()]),color='k')
plt.title('E-W l2-model.')

plt.xlabel('x')
plt.ylabel('z')
ax3.set_xlim(-60,60)
ax3.set_ylim(-80,0)  
ax3.set_aspect('equal') 
ax3.axis('equal')                                                
plt.colorbar(im3[0],ax=ax3, orientation='horizontal')

