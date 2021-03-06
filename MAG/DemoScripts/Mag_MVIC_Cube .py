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
import scipy.sparse as sp

# # STEP 1: Setup and data simulation # #

# Magnetic inducing field parameter (A,I,D)
B = [50000, 90, 0]

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
survey_amp = PF.BaseMag.LinearSurvey(srcField)

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

# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

# Create reduced identity map
idenMap = Maps.IdentityMap(nP=nC)

# Create the forward problem (forwardOnly)
prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap, actInd=actv,
                                     M=M, silent=True)

# Pair the survey and problem
survey.pair(prob)


prob.forwardOnly = True
pred_x = prob.Intrgl_Fwr_Op(m=model, recType='x')
pred_y = prob.Intrgl_Fwr_Op(m=model, recType='y')
pred_z = prob.Intrgl_Fwr_Op(m=model, recType='z')
d_TMI = prob.Intrgl_Fwr_Op(m=model, recType='tmi')

ndata = survey.nD

d_amp = np.sqrt(pred_x**2. +
                pred_y**2. +
                pred_z**2.)

# Add noise and uncertainties
# We add some random Gaussian noise (1nT)
d_TMI += np.random.randn(len(d_TMI))*0.
wd = np.ones(len(d_TMI))*1.  # Assign flat uncertainties
survey.dobs = d_TMI
survey.std = wd

rxLoc = survey.srcField.rxList[0].locs

# # RUN THE MVI-CARTESIAN
# Create identity map
idenMap = Maps.IdentityMap(nP=4*nC)

mstart = np.ones(3*len(actv))*1e-4

# Create a block diagonal misfit wire
wires_misfit = Maps.Wires(('pst', 3*nC))

# Create the forward model operator
prob_MVI = PF.Magnetics.MagneticVector(mesh, chiMap=wires_misfit.pst,
                                     actInd=actv, silent=True)

# Explicitely set starting model
prob_MVI.model = mstart

# Pair the survey and problem
survey.pair(prob_MVI)

dmis_MVI = DataMisfit.l2_DataMisfit(survey)
dmis_MVI.W = 1./survey.std


# Create a block diagonal regularization
wires = Maps.Wires(('p', nC), ('s', nC), ('t', nC))

# Create a regularization
# Create a regularization
reg_p = Regularization.Sparse(mesh, indActive=actv, mapping=wires.p)
reg_p.norms = [2, 2, 2, 2]

reg_s = Regularization.Sparse(mesh, indActive=actv, mapping=wires.s)
reg_s.norms = [2, 2, 2, 2]

reg_t = Regularization.Sparse(mesh, indActive=actv, mapping=wires.t)
reg_t.norms = [2, 2, 2, 2]

reg = reg_p + reg_s + reg_t
reg.mref = np.zeros(4*nC)


dmis = dmis_MVI

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=10,
                                 lower=[-np.inf],
                                 upper=[np.inf],
                                 maxIterLS=10,
                                 maxIterCG=20, tolCG=1e-3,
                                 LSalwaysPass=True,
                                 stepOffBoundsFact=1e-8)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

# LIST OF DIRECTIVES
# betaest = Directives.BetaEstimate_ByEig()
IRLS = Directives.Update_IRLS(f_min_change=1e-4,
                              minGNiter=1, beta_tol=1e-2,
                              coolingRate=1)
update_SensWeight = Directives.UpdateSensWeighting()
update_Jacobi = Directives.UpdatePreCond()
ProjSpherical = Directives.ProjSpherical()
JointAmpMVI = Directives.JointAmpMVI()
betaest = Directives.BetaEstimate_ByEig()
#saveModel = Directives.SaveUBCVectorsEveryIteration(mapping=actvMap,
#                                                    saveComp=True,
#                                                    spherical=True)

inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, IRLS, update_SensWeight, 
                                             update_Jacobi])

# Run JOINT

mrec = inv.run(mstart)

#NOTE - Would like to have dpred working on both surveys

dpred = invProb.getFields(mrec)
#%% Plot models
from matplotlib.patches import Rectangle

contours = [0.01]
xlim = [-60,60]

ypanel = midx
zpanel = -4

fig = plt.figure(figsize=(5, 5))
ax3 = plt.subplot(2,2,1)
out = PF.Magnetics.plot_obs_2D(rxLoc, d=d_TMI, fig=fig, ax=ax3)
ax3.set_title('Obs-TMI')
ax3.set_xlabel('X (m)')
ax3.set_ylabel('Y (m)')

#ax2 = plt.subplot(2,2,2)
#out = PF.Magnetics.plot_obs_2D(rxLoc, d=d_amp, fig=fig, ax=ax2)
#ax3.set_title('Obs-Amplitude')
#ax3.set_xlabel('X (m)')
#ax3.set_ylabel('Y (m)')

ax2 = plt.subplot(2,2,3)
out = PF.Magnetics.plot_obs_2D(rxLoc, d=d_TMI-dpred, fig=fig, ax=ax2)
ax3.set_title('Res-TMI')
ax3.set_xlabel('X (m)')
ax3.set_ylabel('Y (m)')

#ax2 = plt.subplot(2,2,4)
#out = PF.Magnetics.plot_obs_2D(rxLoc, d=d_amp-dpred[0], fig=fig, ax=ax2)
#ax3.set_title('Res-Amplitude')
#ax3.set_xlabel('X (m)')
#ax3.set_ylabel('Y (m)')

#fig = plt.figure(figsize=(5, 2.5))
#ax2 = plt.subplot()
#PF.Magnetics.plotModelSections(mesh, model, normal='y', ind=midy, subFact=2, scale=0.25, xlim=[-xlim, xlim], ylim=[-xlim, 5],
#                      axs=ax2, vmin=0, vmax=vmax, contours = contours)
#for midx in locx:
#    ax2.add_patch(Rectangle((mesh.vectorCCx[midx-nX]-dx/2.,mesh.vectorCCz[midz-nX]-dx/2.),(2*nX+1)*dx,(2*nX+1)*dx, facecolor = 'none', edgecolor='k'))
#ax2.grid(color='w', linestyle='--', linewidth=0.5)
#loc = ax2.get_position()
#ax2.set_position([loc.x0+0.025, loc.y0+0.025, loc.width, loc.height])
#ax2.set_xlabel('X (m)')
#ax2.set_ylabel('Depth (m)')

vmax = None
vmin = model.min()
fig = plt.figure(figsize=(5, 2.5))
ax2 = plt.subplot()
scl_vec = np.max(mrec)/np.max(model) * 0.2
PF.Magnetics.plotModelSections(mesh, wires_misfit.pst*mrec, normal='y',
                               ind=ypanel, axs=ax2,
                               xlim=xlim, scale=scl_vec, vec='w',
                               ylim=[xlim[0], 5],
                               vmin=vmin, vmax=vmax)
for midx in locx:
    ax2.add_patch(Rectangle((mesh.vectorCCx[midx-nX]-dx/2.,mesh.vectorCCz[midz-nX]-dx/2.),(2*nX+1)*dx,(2*nX+1)*dx, facecolor = 'none', edgecolor='k'))
ax2.grid(color='w', linestyle='--', linewidth=0.5)


loc = ax2.get_position()
ax2.set_position([loc.x0+0.025, loc.y0+0.025, loc.width, loc.height])
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Depth (m)')
ax2.set_title('MVI solution')



loc = ax2.get_position()
ax2.set_position([loc.x0+0.025, loc.y0+0.025, loc.width, loc.height])
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Depth (m)')
ax2.set_title('MVI-C Section')


vmax = None
vmin = model.min()
fig = plt.figure(figsize=(5, 5))
ax2 = plt.subplot()
scl_vec = np.max(mrec)/np.max(model) * 0.2
PF.Magnetics.plotModelSections(mesh, wires_misfit.pst*mrec, normal='z',
                               ind=-3, axs=ax2,
                               xlim=xlim, scale=scl_vec, vec='w',
                               ylim=xlim,
                               vmin=vmin, vmax=vmax)
ax2.set_title('Plan view')
ax2.set_ylabel('Elevation (m)', size=14)
