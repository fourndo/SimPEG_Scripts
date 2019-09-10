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
model[(midx-1):(midx+1), (midy-1):(midy+1), -6:-3] = 0.1
model = Utils.mkvc(model)
model = np.asarray(np.r_[np.zeros(nC),np.zeros(nC),model[actv]])

# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

# Creat reduced identity map
idenMap = Maps.IdentityMap(nP=3*nC)

# Create the forward model operator
prob = PF.Magnetics.MagneticVector(mesh, chiMap=idenMap,
                                     actInd=actv)

# Pair the survey and problem
survey.pair(prob)

# Compute linear forward operator and compute some data
d = prob.fields(model)

#PF.Magnetics.plot_obs_2D(survey.srcField.rxList[0].locs,d)

# Add noise and uncertainties (1nT)
data = d + np.random.randn(len(d))
wd = np.ones(len(data))*1.

survey.dobs = data
survey.std = wd

# Create sensitivity weights from our linear forward operator
wr = np.sum(prob.G**2., axis=0)**0.5
wr = (wr/np.max(wr))

# Create a regularization
reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap, nSpace=3)
reg.cell_weights = wr
reg.mref = np.zeros(3*nC)

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = 1/wd

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=30,lower=-10.,upper=10., maxIterCG= 20, tolCG = 1e-3)


invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
betaest = Directives.BetaEstimate_ByEig()

# Here is where the norms are applied
IRLS = Directives.Update_IRLS(norms=([2, 2, 2, 2]),
                              eps=None, f_min_change=1e-4,
                              minGNiter=3, beta_tol=1e-2)

update_Jacobi = Directives.Update_lin_PreCond()

inv = Inversion.BaseInversion(invProb,
                              directiveList=[update_Jacobi,IRLS, betaest])
                                                  
mrec = inv.run(np.ones(3*len(actv))*1e-4)

ypanel = midx
zpanel = -4


#m_l2 = actvMap * reg.l2model[0:nC]
#m_l2[m_l2==-100] = np.nan

m_lpx = actvMap * mrec[0:nC]
m_lpy = actvMap * mrec[nC:2*nC]
m_lpz = actvMap * -mrec[2*nC:]

m_lpx[m_lpx==-100] = np.nan
m_lpy[m_lpy==-100] = np.nan
m_lpz[m_lpz==-100] = np.nan

amp = np.sqrt(m_lpx**2. + m_lpy**2. + m_lpz**2.)

m_lpx = (m_lpx/amp).reshape(mesh.vnC, order='F')
m_lpy = (m_lpy/amp).reshape(mesh.vnC, order='F')
m_lpz = (m_lpz/amp).reshape(mesh.vnC, order='F')

sub = 2
#%% Plot the result
fig = plt.figure()
ax2 = plt.subplot(221)
ax3 = plt.subplot(222)

xx, yy = mesh.gridCC[:,0].reshape(mesh.vnC, order="F"), mesh.gridCC[:,1].reshape(mesh.vnC, order="F")
zz = mesh.gridCC[:,2].reshape(mesh.vnC, order="F")



amp = amp.reshape(mesh.vnC, order='F')
#ptemp = ma.array(ptemp ,mask=np.isnan(ptemp))
ax2.contourf(xx[:,:,zpanel].T,yy[:,:,zpanel].T,amp[:,:,zpanel],20)
im2 = ax2.quiver(mkvc(xx[::sub,::sub,zpanel]),mkvc(yy[::sub,::sub,zpanel]),
                 mkvc(m_lpx[::sub,::sub,zpanel]),mkvc(m_lpy[::sub,::sub,zpanel]),pivot='mid',
                 units="xy", scale=0.2, linewidths=(1,), edgecolors=('k'), headaxislength=0.1,
                    headwidth = 10, headlength=30)


           
#im2 = mesh.plotSlice(m_lp, ax = ax2, normal = 'Z', ind=zpanel, grid=True,pcolorOpts={'cmap':'viridis'})
#plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCy[ypanel],mesh.vectorCCy[ypanel]]),color='w')
plt.title('Plan l2-model.')
ax2.set_aspect('equal')
plt.ylabel('y')
ax2.xaxis.set_visible(False)
ax2.set_xlim(-60,60)
ax2.set_ylim(-60,60)    
ax2.axis('equal')

im2 = ax3.contourf(xx[:,ypanel,:],zz[:,ypanel,:],amp[:,ypanel,:],20)
ax3.quiver(mkvc(xx[::sub,ypanel,::sub]),mkvc(zz[::sub,ypanel,::sub]),
                 mkvc(m_lpx[::sub,ypanel,::sub]),mkvc(m_lpz[::sub,ypanel,::sub]),pivot='mid',
                 units="xy", scale=0.2, linewidths=(1,), edgecolors=('k'), headaxislength=0.1,
                    headwidth = 10, headlength=30)
#plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCz[zpanel],mesh.vectorCCz[zpanel]]),color='w')
#plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([Z.min(),Z.max()]),color='k')
plt.title('E-W l2-model.')

plt.xlabel('x')
plt.ylabel('z')
ax3.set_xlim(-60,60)
ax3.set_ylim(-80,0)  
ax3.set_aspect('equal') 
ax3.axis('equal')                                                
plt.colorbar(im2,ax=ax3)
