"""

This script runs an Magnetic Amplitude Inversion (MAI) from TMI data.
Magnetic amplitude data are weakly sensitive to the orientation of
magnetization, and can therefore better recover the location and geometry of
magnetic bodies in the presence of remanence. The algorithm is inspired from
Li & Shearer (2008), with an added iterative sensitivity weighting strategy to
counter the vertical streatchin that thpe old code had

This is done in three parts:

1- TMI data are inverted for an equivalent source layer.

2-The equivalent source layer is used to predict component data -> amplitude

3- Amplitude data are inverted in 3-D for an effective susceptibility model

Created on December 7th, 2016

@author: fourndo@gmail.com

"""
from SimPEG import Mesh, Directives, Maps, InvProblem, Optimization, DataMisfit, Inversion, Utils, Regularization
import SimPEG.PF as PF
from SimPEG import mkvc
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import os

# # STEP 1: Setup and data simulation # #


# Magnetic inducing field parameter (A,I,D)
B = [50000, 90, 0]

from SimPEG import Mesh
# Create a mesh
dx = 5.
npad = 0
hxind = [(dx, npad, -1.3), (dx, 24), (dx, npad, 1.3)]
hyind = [(dx, npad, -1.3), (dx, 24), (dx, npad, 1.3)]
hzind = [(dx, 0, -1.3), (dx, 12)]

mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CC0')
mesh.x0[2] -= mesh.vectorNz[-1]

susc = 0.025
nX = 2

# Get index of the center of block
locx = [int(mesh.nCx/2)]#[int(mesh.nCx/2)-3, int(mesh.nCx/2)+3]
midy = int(mesh.nCy/2)
midz = -5

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
xr = np.linspace(-25., 25., 20)
yr = np.linspace(-25., 25., 20)
X, Y = np.meshgrid(xr, yr)

# Move the observation points 5m above the topo
Z = np.ones_like(X) * mesh.vectorNz[-1] + dx

# Create a MAGsurvey
rxLoc = np.c_[Utils.mkvc(X.T), Utils.mkvc(Y.T), Utils.mkvc(Z.T)]
rxLoc = PF.BaseMag.RxObs(rxLoc)
srcField = PF.BaseMag.SrcField([rxLoc], param=(B[0], B[1], B[2]))
survey = PF.BaseMag.LinearSurvey(srcField)

# We can now create a susceptibility model and generate data
# Here a simple block in half-space
model = np.zeros((mesh.nCx, mesh.nCy, mesh.nCz))
for midx in locx:
    model[(midx-nX):(midx+nX+1), (midy-nX):(midy+nX+1), (midz-nX):(midz+nX+1)] = susc
model = Utils.mkvc(model)
model = model[actv]

# We create a magnetization model different than the inducing field
# to simulate remanent magnetization. Let's do something simple [45,90]
M = PF.Magnetics.dipazm_2_xyz(np.ones(nC) * 0., np.ones(nC) * 90.)
#M = PF.Magnetics.dipazm_2_xyz(np.ones(nC) * B[1], np.ones(nC) * B[2])


m = mkvc(sp.diags(model, 0) * M)

# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

# Create reduced identity map
idenMap = Maps.IdentityMap(nP=nC)

# Create the forward problem (forwardOnly)
prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap, actInd=actv,
                                     M=M, forwardOnly=True)

# Pair the survey and problem
survey.pair(prob)

# Compute forward model some data
d = prob.fields(model)

# Add noise and uncertainties
# We add some random Gaussian noise (1nT)
d_TMI = d + np.random.randn(len(d))*0.
wd = np.ones(len(d_TMI))  # Assign flat uncertainties
survey.dobs = d_TMI
survey.std = wd

# %% # RUN MVI INVERSION
# Create active map to go from reduce set to full
# Creat reduced identity map
idenMap = Maps.IdentityMap(nP=3*nC)

# Create the forward model operator
prob = PF.Magnetics.MagneticVector(mesh, chiMap=idenMap,
                                     actInd=actv, silent=True)


# Set starting mdoel
mstart = np.ones(3*len(actv))*1e-4
# Explicitely set starting model
prob.model = mstart


# Pair the survey and problem
survey.pair(prob)


# Create sensitivity weights from our linear forward operator
wr = np.sum(prob.F**2., axis=0)**0.5
wr = (wr/np.max(wr))

#prob.JtJdiag = wr
# Create a block diagonal regularization
wires = Maps.Wires(('p', nC), ('s', nC), ('t', nC))

# Create a regularization
reg_p = Regularization.Sparse(mesh, indActive=actv, mapping=wires.p)
reg_p.cell_weights = wires.p * wr
reg_p.norms = [2, 2, 2, 2]

reg_s = Regularization.Sparse(mesh, indActive=actv, mapping=wires.s)
reg_s.cell_weights = wires.s * wr
reg_s.norms = [2, 2, 2, 2]

reg_t = Regularization.Sparse(mesh, indActive=actv, mapping=wires.t)
reg_t.cell_weights = wires.t * wr
reg_t.norms = [2, 2, 2, 2]

reg = reg_p + reg_s + reg_t
reg.mref = np.zeros(3*nC)

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=10, lower=-10., upper=10.,
                                 maxIterCG=20, tolCG=1e-3)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
betaest = Directives.BetaEstimate_ByEig()

# Here is where the norms are applied
IRLS = Directives.Update_IRLS(f_min_change=1e-4,
                              minGNiter=3, beta_tol=1e-2)
# IRLS._target = 10
update_SensWeight = Directives.UpdateSensWeighting()
update_Jacobi = Directives.UpdatePreCond()
targetMisfit = Directives.TargetMisfit()

# saveModel = Directives.SaveUBCVectorsEveryIteration(mapping=actvMap)
# saveModel.fileName = work_dir + out_dir + 'MVI_C'
inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, IRLS, update_Jacobi])

mrec_MVI = inv.run(mstart)

beta = invProb.beta

# %% RUN MVI-S

# # STEP 3: Finish inversion with spherical formulation
mstart = PF.Magnetics.xyz2atp(mrec_MVI)

prob.coordinate_system = 'spherical'
prob.model = mstart

# Create a block diagonal regularization
wires = Maps.Wires(('amp', nC), ('theta', nC), ('phi', nC))

# Create a regularization
reg_a = Regularization.Sparse(mesh, indActive=actv, mapping=wires.amp)
reg_a.norms = [0,1,1,1]
reg_a.alpha_s = 1
reg_a.eps_p = 1e-4
reg_a.eps_q = 1e-4


reg_t = Regularization.Sparse(mesh, indActive=actv, mapping=wires.theta)
reg_t.alpha_s = 0.
reg_t.space = 'spherical'
reg_t.norms = [2, 0,0,0]
reg_t.eps_q = 2e-1
# reg_t.alpha_x, reg_t.alpha_y, reg_t.alpha_z = 0.25, 0.25, 0.25

reg_p = Regularization.Sparse(mesh, indActive=actv, mapping=wires.phi)
reg_p.alpha_s = 0.
reg_p.space = 'spherical'
reg_p.norms = [2,0,0,0]
reg_p.eps_q = 1e-1

reg = reg_a + reg_t + reg_p
reg.mref = np.zeros(3*nC)

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std


Lbound = np.kron(np.asarray([0, -np.inf, -np.inf]),np.ones(nC))
Ubound = np.kron(np.asarray([10, np.inf, np.inf]),np.ones(nC))

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=25,
                                 lower=Lbound,
                                 upper=Ubound,
                                 maxIterLS=10,
                                 maxIterCG=20, tolCG=1e-3,
                                 stepOffBoundsFact=1e-8)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=beta*10.)
#  betaest = Directives.BetaEstimate_ByEig()

# Here is where the norms are applied
IRLS = Directives.Update_IRLS(f_min_change=1e-4,
                              minGNiter=2, beta_tol=1e-2,
                              coolingRate=2)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=beta)
# betaest = Directives.BetaEstimate_ByEig()


# Special directive specific to the mag amplitude problem. The sensitivity
# weights are update between each iteration.
update_SensWeight = Directives.UpdateSensWeighting()
update_Jacobi = Directives.UpdatePreCond()
ProjSpherical = Directives.ProjSpherical()
betaest = Directives.BetaEstimate_ByEig()



inv = Inversion.BaseInversion(invProb,
                              directiveList=[ProjSpherical, IRLS, update_SensWeight,  
                                             update_Jacobi])


mrec_MVI_S = inv.run(mstart)

mrec_MVI_S_xyz = PF.Magnetics.atp2xyz(mrec_MVI_S)

#%% Plot models
from matplotlib.patches import Rectangle

contours = [0.02]

vmin = 0.
xlim = [-60, 60]

# # FIRST MODEL # #
fig = plt.figure(figsize=(5, 2.5))
ax2 = plt.subplot()

vmax = model.max()
ax2, im2, cbar = PF.Magnetics.plotModelSections(mesh, m, normal='y',
                               ind=midy, axs=ax2,
                               xlim=xlim, scale = 0.3, vec ='w',
                               ylim=[xlim[0], 5],
                               vmin=vmin, vmax=vmax)
for midx in locx:
    ax2.add_patch(Rectangle((mesh.vectorCCx[midx-nX]-dx/2.,mesh.vectorCCz[midz-nX]-dx/2.),(2*nX+1)*dx,(2*nX+1)*dx, facecolor = 'none', edgecolor='k'))
ax2.grid(color='w', linestyle='--', linewidth=0.5)
loc = ax2.get_position()
ax2.set_position([loc.x0+0.025, loc.y0+0.025, loc.width, loc.height])
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Depth (m)')

# fig = plt.figure(figsize=(5, 2.5))
# ax1 = plt.subplot()

# ax1, im2, cbar = PF.Magnetics.plotModelSections(mesh, m, normal='z',
#                                ind=midz, axs=ax1,
#                                xlim=xlim, scale = 0.3, vec ='w',
#                                ylim=xlim,
#                                vmin=vmin, vmax=vmax)
# for midx in locx:
#     ax1.add_patch(Rectangle((mesh.vectorCCx[midx-nX-1]+dx/2.,mesh.vectorCCy[midy-nX-1]+dx/2.),3*dx,3*dx, lw=2, facecolor = 'none', edgecolor='r'))
# ax1.set_title('(a) True')
# ax1.xaxis.set_visible(False)

# # SECOND MODEL # #
vmax = mrec_MVI.max()
scale = mrec_MVI.max()/m.max()*0.3
fig = plt.figure(figsize=(5, 2.5))
ax2 = plt.subplot()
# ax3, im2, cbar = PF.Magnetics.plotModelSections(mesh, mrec_MVI, normal='z',
#                                ind=midz, axs=ax3,
#                                xlim=xlim, scale = scale, vec ='w',
#                                ylim=xlim,
#                                vmin=vmin, vmax=vmax)
# for midx in locx:
#     ax3.add_patch(Rectangle((mesh.vectorCCx[midx-nX-1]+dx/2.,mesh.vectorCCy[midy-nX-1]+dx/2.),3*dx,3*dx, lw=2, facecolor = 'none', edgecolor='r'))

ax2, im2, cbar = PF.Magnetics.plotModelSections(mesh, mrec_MVI, normal='y',
                               ind=midy, axs=ax2,
                               xlim=xlim, scale = scale, vec ='w',
                               ylim=[xlim[0], 5],
                               vmin=vmin, vmax=vmax)
for midx in locx:
    ax2.add_patch(Rectangle((mesh.vectorCCx[midx-nX]-dx/2.,mesh.vectorCCz[midz-nX]-dx/2.),(2*nX+1)*dx,(2*nX+1)*dx, facecolor = 'none', edgecolor='k'))
ax2.grid(color='w', linestyle='--', linewidth=0.5)
loc = ax2.get_position()
ax2.set_position([loc.x0+0.025, loc.y0+0.025, loc.width, loc.height])
ax2.set_title('MVI-C')
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Depth (m)')

# # THIRD MODEL # #
fig = plt.figure(figsize=(5, 2.5))
ax2 = plt.subplot()
vmax = mrec_MVI_S_xyz.max()
scale = mrec_MVI_S_xyz.max()/m.max()*0.2

# # ax5, im2, cbar = PF.Magnetics.plotModelSections(mesh, mrec_MVI_S, normal='z',
# #                                ind=midz, axs=ax5,
# #                                xlim=xlim, scale = scale, vec ='w',
# #                                ylim=xlim,
# #                                vmin=vmin, vmax=vmax)
# for midx in locx:
#     ax5.add_patch(Rectangle((mesh.vectorCCx[midx-nX-1]+dx/2.,mesh.vectorCCy[midy-nX-1]+dx/2.),3*dx,3*dx, lw=2, facecolor = 'none', edgecolor='r'))


ax2, im2, cbar = PF.Magnetics.plotModelSections(mesh, mrec_MVI_S_xyz, normal='y',
                               ind=midy, axs=ax2,
                               xlim=xlim, scale = scale, vec ='w',
                               ylim=[xlim[0], 5],
                               vmin=vmin, vmax=vmax)
for midx in locx:
    ax2.add_patch(Rectangle((mesh.vectorCCx[midx-nX]-dx/2.,mesh.vectorCCz[midz-nX]-dx/2.),(2*nX+1)*dx,(2*nX+1)*dx, facecolor = 'none', edgecolor='k'))
ax2.grid(color='w', linestyle='--', linewidth=0.5)
loc = ax2.get_position()
ax2.set_position([loc.x0+0.025, loc.y0+0.025, loc.width, loc.height])
ax2.set_title('MVI-S')
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Depth (m)')

plt.show()
