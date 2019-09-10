import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from numpy import pi as pi

from SimPEG import Mesh
from SimPEG import Utils
from SimPEG import Maps
from SimPEG import Regularization
from SimPEG import DataMisfit
from SimPEG import Optimization
from SimPEG import InvProblem
from SimPEG import Directives
from SimPEG import Inversion
from SimPEG import PF
from SimPEG import mkvc
import os

out_dir = 'C:\Users\dominiquef.MIRAGEOSCIENCE\ownCloud\Research\Modelling\Synthetic\SingleBlock\Simpeg'
#def run(plotIt=True):
"""
    PF: Magnetics Vector Inversion - Spherical
    ==========================================

    In this example, we invert for the 3-component magnetization vector
    with the Spherical formulation. The code is used to invert magnetic
    data affected by remanent magnetization and makes no induced
    assumption. The inverse problem is highly non-linear and has proven to
    be challenging to solve. We introduce an iterative sensitivity
    weighting to improve the solution. The spherical formulation allows for
    compact norms to be applied on the magnitude and direction of
    magnetization independantly, hence reducing the complexity over
    the usual smooth MVI solution.

    The algorithm builds upon the research done at UBC:

    Lelievre, G.P., 2009, Integrating geological and geophysical data
    through advanced constrained inversions. PhD Thesis, UBC-GIF

    The steps are:
    1- STEP 1: Create a synthetic model and calculate TMI data. This will
    simulate the usual magnetic experiment.

    2- STEP 2: Invert for a starting model with cartesian formulation.

    3- STEP 3: Invert for a compact mag model with spherical formulation.

"""

# # STEP 1: Setup and data simulation # #

# Magnetic inducing field parameter (A,I,D)
B = [50000, 90, 0]

# Create a mesh
dx = 5.

hxind = [(dx, 5, -1.3), (dx, 15), (dx, 5, 1.3)]
hyind = [(dx, 5, -1.3), (dx, 15), (dx, 5, 1.3)]
hzind = [(dx, 5, -1.3), (dx, 7)]

mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCC')

# Get index of the center
midx = int(mesh.nCx/2)
midy = int(mesh.nCy/2)

# Lets create a simple flat topo and set the active cells
[xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
zz = np.ones_like(xx)*mesh.vectorNz[-1]
topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]

# Go from topo to actv cells
actv = Utils.surface2ind_topo(mesh, topo, 'N')
actv = np.asarray([inds for inds, elem in enumerate(actv, 1) if elem],
                  dtype=int) - 1

# Create active map to go from reduce space to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)
nC = int(len(actv))

# Create and array of observation points
xr = np.linspace(-30., 30., 20)
yr = np.linspace(-30., 30., 20)
X, Y = np.meshgrid(xr, yr)

# Move the observation points 5m above the topo
Z = np.ones_like(X) * mesh.vectorNz[-1] + dx

# Create a MAGsurvey
rxLoc = np.c_[Utils.mkvc(X.T), Utils.mkvc(Y.T), Utils.mkvc(Z.T)]
rxObj = PF.BaseMag.RxObs(rxLoc)
srcField = PF.BaseMag.SrcField([rxObj], param=(B[0], B[1], B[2]))
survey = PF.BaseMag.LinearSurvey(srcField)

# We can now create a susceptibility model and generate data
# Here a simple block in half-space
model = np.zeros((mesh.nCx, mesh.nCy, mesh.nCz))
model[(midx-2):(midx+2), (midy-2):(midy+2), -6:-2] = 0.05
model = Utils.mkvc(model)
model = model[actv]

# We create a magnetization model different than the inducing field
# to simulate remanent magnetization. Let's do something simple,
# reversely magnetized [45,90]
M = PF.Magnetics.dipazm_2_xyz(np.ones(nC) * 45., np.ones(nC) * 90.)

# Multiply the orientation with the effective susceptibility
# and reshape as [mx,my,mz] vector
m = mkvc(sp.diags(model, 0) * M)

# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

# Create reduced identity map
idenMap = Maps.IdentityMap(nP=3*nC)

# Create the forward model operator
prob = PF.Magnetics.MagneticVector(mesh, chiMap=idenMap,
                                   actInd=actv)

# Pair the survey and problem
survey.pair(prob)

# Compute forward model some data
d = prob.fields(m)

# Add noise and uncertainties
# We add some random Gaussian noise (1nT)
d_TMI = d + np.random.randn(len(d))*0.
wd = np.ones(len(d_TMI))  # Assign flat uncertainties
survey.dobs = d_TMI
survey.std = wd

# # STEP 2: Invert for a magnetization model in Cartesian space # #

# Create a static sensitivity weighting function
wr = np.sum(prob.G**2., axis=0)**0.5
wr = (wr/np.max(wr))

# Create a block diagonal regularization
reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap,
                            nSpace=3)
reg.cell_weights = wr
reg.mref = np.zeros(3*nC)

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = 1./survey.std

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=10, lower=-10., upper=10.,
                                 maxIterCG=20, tolCG=1e-3)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
betaest = Directives.BetaEstimate_ByEig()

# Here is where the norms are applied
IRLS = Directives.Update_IRLS(norms=([2, 2, 2, 2]),
                              eps=None, f_min_change=1e-4,
                              minGNiter=3, beta_tol=1e-2)

update_Jacobi = Directives.Update_lin_PreCond()

inv = Inversion.BaseInversion(invProb,
                              directiveList=[ IRLS, update_Jacobi,betaest])

mstart = np.ones(3*nC)*1e-4
mrec_C = inv.run(mstart)

beta = invProb.beta*1.

# # STEP 3: Finish inversion with spherical formulation
mstart = PF.Magnetics.xyz2atp(mrec_C)
prob.ptype = 'Spherical'
prob.chi = mstart

# Create a block diagonal regularization
reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap,
                            nSpace=3)
reg.mref = np.zeros(3*nC)
reg.cell_weights = np.ones(3*nC)
reg.alpha_s = [1., 0., 0.]
reg.mspace = ['lin','sph','sph']

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = 1./survey.std

# Add directives to the inversion
opt = Optimization.ProjectedGNCG_nSpace(maxIter=20, lower=[0.,-pi/2.,-pi],
                                        upper=[10.,pi/2.,pi], maxIterLS = 5,
                                        LSreduction = 1e-1,
                                        maxIterCG=40, tolCG=1e-3,
                                        ptype = ['lin','sph','sph'], nSpace=3)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta = beta)
#betaest = Directives.BetaEstimate_ByEig()

# Here is where the norms are applied
IRLS = Directives.Update_IRLS(norms=([0, 1, 1, 1]),
                              eps=None, f_min_change=1e-4,
                              minGNiter=3, beta_tol=1e-2,
                              coolingRate = 3)

# Special directive specific to the mag amplitude problem. The sensitivity
# weights are update between each iteration.
update_Jacobi = Directives.Amplitude_Inv_Iter()
update_Jacobi.ptype = 'MVI-S'

inv = Inversion.BaseInversion(invProb,
                              directiveList=[IRLS, update_Jacobi, ])



mrec = inv.run(mstart)

#    if plotIt:
# Here is the recovered susceptibility model
ypanel = midx
zpanel = -4

vmin = model.min()
vmax = model.max()/5.

m_lpx = actvMap * mrec_C[0:nC]
m_lpy = actvMap * mrec_C[nC:2*nC]
m_lpz = actvMap * -mrec_C[2*nC:]

m_lpx[m_lpx == -100] = np.nan
m_lpy[m_lpy == -100] = np.nan
m_lpz[m_lpz == -100] = np.nan

amp = np.sqrt(m_lpx**2. + m_lpy**2. + m_lpz**2.)

m_lpx = (m_lpx/amp).reshape(mesh.vnC, order='F')
m_lpy = (m_lpy/amp).reshape(mesh.vnC, order='F')
m_lpz = (m_lpz/amp).reshape(mesh.vnC, order='F')
amp = amp.reshape(mesh.vnC, order='F')

sub = 2

xx = mesh.gridCC[:, 0].reshape(mesh.vnC, order="F")
zz = mesh.gridCC[:, 2].reshape(mesh.vnC, order="F")
yy = mesh.gridCC[:, 1].reshape(mesh.vnC, order="F")

fig = plt.figure(figsize=(8, 8))
ax2 = plt.subplot(312)
im2 = ax2.contourf(xx[:, ypanel, :].T, zz[:, ypanel, :].T,
                   amp[:, ypanel, :].T, 40,
                   vmin=vmin, vmax=vmax, clim=[vmin, vmax],
                   cmap='magma_r')

ax2.quiver(mkvc(xx[::sub, ypanel, ::sub].T),
           mkvc(zz[::sub, ypanel, ::sub].T),
           mkvc(m_lpx[::sub, ypanel, ::sub].T),
           mkvc(m_lpz[::sub, ypanel, ::sub].T), pivot='mid',
           units="xy", scale=0.2, linewidths=(1,), edgecolors=('k'),
           headaxislength=0.1, headwidth=10, headlength=30)
plt.colorbar(im2, orientation="vertical", ax=ax2,
             ticks=np.linspace(im2.vmin, im2.vmax, 4),
             format="${%.3f}$")

ax2.set_aspect('equal')
ax2.set_xlim(-50, 50)
ax2.set_ylim(mesh.vectorNz[3], mesh.vectorNz[-1]+dx)
ax2.set_title('EW - Recovered')
ax2.set_xlabel('Easting (m)', size=14)
ax2.set_ylabel('Elevation (m)', size=14)


mrec = PF.Magnetics.atp2xyz(mrec)

m_lpx = actvMap * mrec[0:nC]
m_lpy = actvMap * mrec[nC:2*nC]
m_lpz = actvMap * -mrec[2*nC:]

m_lpx[m_lpx == -100] = np.nan
m_lpy[m_lpy == -100] = np.nan
m_lpz[m_lpz == -100] = np.nan

M_out = np.c_[m_lpx, m_lpy, -m_lpz]
Mesh.TensorMesh.writeVectorUBC(mesh,out_dir + '\Recovered.vec',M_out)
amp = np.sqrt(m_lpx**2. + m_lpy**2. + m_lpz**2.)+1e-8

m_lpx = (m_lpx/amp).reshape(mesh.vnC, order='F')
m_lpy = (m_lpy/amp).reshape(mesh.vnC, order='F')
m_lpz = (m_lpz/amp).reshape(mesh.vnC, order='F')
amp = amp.reshape(mesh.vnC, order='F')

sub = 2

xx = mesh.gridCC[:, 0].reshape(mesh.vnC, order="F")
zz = mesh.gridCC[:, 2].reshape(mesh.vnC, order="F")
yy = mesh.gridCC[:, 1].reshape(mesh.vnC, order="F")


ax2 = plt.subplot(313)
im2 = ax2.contourf(xx[:, ypanel, :].T, zz[:, ypanel, :].T,
                   amp[:, ypanel, :].T, 40,
                   vmin=vmin, vmax=vmax, clim=[vmin, vmax],
                   cmap='magma_r')

ax2.quiver(mkvc(xx[::sub, ypanel, ::sub].T),
           mkvc(zz[::sub, ypanel, ::sub].T),
           mkvc(m_lpx[::sub, ypanel, ::sub].T),
           mkvc(m_lpz[::sub, ypanel, ::sub].T), pivot='mid',
           units="xy", scale=0.2, linewidths=(1,), edgecolors=('k'),
           headaxislength=0.1, headwidth=10, headlength=30)
plt.colorbar(im2, orientation="vertical", ax=ax2,
             ticks=np.linspace(im2.vmin, im2.vmax, 4),
             format="${%.3f}$")

ax2.set_aspect('equal')
ax2.set_xlim(-50, 50)
ax2.set_ylim(mesh.vectorNz[3], mesh.vectorNz[-1]+dx)
ax2.set_title('EW - Recovered Spherical')
ax2.set_xlabel('Easting (m)', size=14)
ax2.set_ylabel('Elevation (m)', size=14)

# plot true model
vmin = model.min()
vmax = model.max()*1.5

m_lpx = (M[:, 0]).reshape(mesh.vnC, order='F')
m_lpy = (M[:, 1]).reshape(mesh.vnC, order='F')
m_lpz = -(M[:, 2]).reshape(mesh.vnC, order='F')
amp = model.reshape(mesh.vnC, order='F')

ax3 = plt.subplot(311)
im2 = ax3.contourf(xx[:, ypanel, :].T, zz[:, ypanel, :].T,
                   amp[:, ypanel, :].T, 40,
                   vmin=vmin, vmax=vmax, clim=[vmin, vmax],
                   cmap='magma_r')

ind = mkvc(amp[::sub, ypanel, ::sub].T) > 0
ax3.quiver(mkvc(xx[::sub, ypanel, ::sub].T)[ind],
           mkvc(zz[::sub, ypanel, ::sub].T)[ind],
           mkvc(m_lpx[::sub, ypanel, ::sub].T)[ind],
           mkvc(m_lpz[::sub, ypanel, ::sub].T)[ind], pivot='mid',
           units="xy", scale=0.2, linewidths=(1,), edgecolors=('k'),
           headaxislength=0.1, headwidth=10, headlength=30)
plt.colorbar(im2, orientation="vertical", ax=ax3,
             ticks=np.linspace(im2.vmin, im2.vmax, 4),
             format="${%.3f}$")
ax3.set_aspect('equal')
ax3.set_xlim(-50, 50)
ax3.set_ylim(mesh.vectorNz[3], mesh.vectorNz[-1]+dx)
ax3.set_title('EW - True')
ax3.xaxis.set_visible(False)
ax3.set_ylabel('Elevation (m)', size=14)

# Plot the data
#fig = plt.figure(figsize=(8, 4))
#ax1 = plt.subplot(121)
#ax2 = plt.subplot(122)
#PF.Magnetics.plot_obs_2D(rxLoc, d=d_TMI, fig=fig, ax=ax1,
#                         varstr='TMI Data')
#PF.Magnetics.plot_obs_2D(rxLoc, d=invProb.dpred, fig=fig, ax=ax2,
#                         varstr='Predicted Data')
#if __name__ == '__main__':
#    run()
#    plt.show()
