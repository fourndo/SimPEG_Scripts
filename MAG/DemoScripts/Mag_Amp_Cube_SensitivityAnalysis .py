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

# # STEP 1: Setup and data simulation # #

# Magnetic inducing field parameter (A,I,D)
B = [50000, 45, 90]

# Create a mesh
dx = 5.
npad = 0
hxind = [(dx, npad, -1.3), (dx, 24), (dx, npad, 1.3)]
hyind = [(dx, npad, -1.3), (dx, 24), (dx, npad, 1.3)]
hzind = [(dx, 0, -1.3), (dx, 20)]

mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CC0')
mesh.x0[2] -= mesh.vectorNz[-1]
norms = [2, 2, 2, 2]
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
Z = np.ones_like(X) * mesh.vectorNz[-1] + 2*dx

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

# For comparison, let's run the inversion assuming an induced response
M = PF.Magnetics.dipazm_2_xyz(np.ones(nC) * B[1], np.ones(nC) * B[2])

# Reset the magnetization
prob.M = M
prob.forwardOnly = False
prob._F = None

# Create a regularization function, in this case l2l2
wr = np.sum(prob.F**2., axis=0)**0.5
wr = (wr/np.max(wr))


# Create a regularization
reg_Susc = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
reg_Susc.cell_weights = wr
reg_Susc.norms= [2, 2, 2, 2]

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
update_Jacobi = Directives.UpdatePreCond()
inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, IRLS, update_Jacobi])

# Run the inversion
m0 = np.ones(nC)*1e-4  # Starting model
mrec_sus = inv.run(m0)
pred_sus = invProb.dpred



# Create active map to go from reduce space to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)
nC = int(len(actv))

# Create identity map
idenMap = Maps.IdentityMap(nP=nC)


# Create the forward model operator
prob = PF.Magnetics.MagneticAmplitude(mesh, chiMap=idenMap,
                                      actInd=actv, M=M, silent=True)

# Define starting model
mstart = np.ones(len(actv))*1e-4
prob.model = mstart



# Change the survey to xyz components
survey.srcField.rxList[0].rxType = 'xyz'

# Pair the survey and problem
survey.unpair()
survey.pair(prob)

# Re-set the observations to |B|
survey.dobs = d_amp

wr = np.sum(prob.F**2., axis=0)**0.5
wr = (wr/np.max(wr))
# Create a sparse regularization
reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
reg.mref = mstart*0.
reg.norms=norms
reg.eps_p = 1e-3
reg.eps_q = 1e-3
reg.cell_weights = wr

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=100, lower=0., upper=10.,
                                 maxIterLS=20, maxIterCG=10,
                                 tolCG=1e-3)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

# Here is the list of directives
betaest = Directives.BetaEstimate_ByEig()

# Specify the sparse norms
IRLS = Directives.Update_IRLS(f_min_change=1e-3,
                              minGNiter=2, chifact=1,
                              coolingRate = 2)

# First experiment we won't update the sensitivity weighting
update_SensWeight = Directives.UpdateSensWeighting(everyIter = False)
update_Jacobi = Directives.UpdatePreCond()

# Put all together
inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, IRLS, update_SensWeight, update_Jacobi, ])

# Invert
mrec_MAI = inv.run(mstart)
pred_MAI = invProb.dpred


reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
reg.mref = mstart*0.
reg.norms=norms
reg.eps_p = 1e-3
reg.eps_q = 1e-3
# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=100, lower=0., upper=10.,
                                 maxIterLS=20, maxIterCG=10,
                                 tolCG=1e-3)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

# Here is the list of directives
betaest = Directives.BetaEstimate_ByEig()

# Specify the sparse norms
IRLS = Directives.Update_IRLS(f_min_change=1e-3,
                              minGNiter=2, chifact=1,
                              coolingRate = 2)

# Special directive specific to the mag amplitude problem. The sensitivity
# weights are update between each iteration.
update_SensWeight = Directives.UpdateSensWeighting()
update_Jacobi = Directives.UpdatePreCond()

# Put all together
inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, IRLS, update_SensWeight, update_Jacobi, ])

# Invert
mrec_MAIS = inv.run(mstart)
pred_MAIS = invProb.dpred


#%% Plot models
from matplotlib.patches import Rectangle

contours = [0.01]
xlim = 60.
fig = plt.figure(figsize=(5, 5))
ax3 = plt.subplot()
vmax = mrec_sus.max()
# PF.Magnetics.plotModelSections(mesh, mrec_sus, normal='z', ind=-3, subFact=2, scale=0.25, xlim=[-xlim, xlim], ylim=[-xlim, xlim],
#                       title="Esus Model", axs=ax3, vmin=0, vmax=vmax, contours = contours)
# ax3.xaxis.set_visible(False)

out = PF.Magnetics.plot_obs_2D(rxLoc, d=pred_sus, fig=fig, ax=ax3,
                               )
ax3.set_xlabel('X (m)')
ax3.set_ylabel('Y (m)')

fig = plt.figure(figsize=(5, 2.5))
ax2 = plt.subplot()
PF.Magnetics.plotModelSections(mesh, model, normal='y', ind=midy, subFact=2, scale=0.25, xlim=[-xlim, xlim], ylim=[-xlim, 5],
                      axs=ax2, vmin=0, vmax=vmax, contours = contours)
for midx in locx:
    ax2.add_patch(Rectangle((mesh.vectorCCx[midx-nX]-dx/2.,mesh.vectorCCz[midz-nX]-dx/2.),(2*nX+1)*dx,(2*nX+1)*dx, facecolor = 'none', edgecolor='k'))
ax2.grid(color='w', linestyle='--', linewidth=0.5)
loc = ax2.get_position()
ax2.set_position([loc.x0+0.025, loc.y0+0.025, loc.width, loc.height])
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Depth (m)')


fig = plt.figure(figsize=(5, 2.5))
ax2 = plt.subplot()
PF.Magnetics.plotModelSections(mesh, mrec_sus, normal='y', ind=midy, subFact=2, scale=0.25, xlim=[-xlim, xlim], ylim=[-xlim, 5],
                      axs=ax2, vmin=0, vmax=vmax, contours = contours)
for midx in locx:
    ax2.add_patch(Rectangle((mesh.vectorCCx[midx-nX]-dx/2.,mesh.vectorCCz[midz-nX]-dx/2.),(2*nX+1)*dx,(2*nX+1)*dx, facecolor = 'none', edgecolor='k'))
ax2.grid(color='w', linestyle='--', linewidth=0.5)
loc = ax2.get_position()
ax2.set_position([loc.x0+0.025, loc.y0+0.025, loc.width, loc.height])
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Depth (m)')


fig = plt.figure(figsize=(5, 5))
ax3 = plt.subplot()
vmax = None#mrec_MAI.max()
# PF.Magnetics.plotModelSections(mesh, mrec_MAI, normal='z', ind=-3, subFact=2, scale=0.25, xlim=[-xlim, xlim], ylim=[-xlim, xlim],
#                       title="Esus Model", axs=ax3, vmin=0, vmax=vmax, contours = contours)
# ax3.xaxis.set_visible(False)

out = PF.Magnetics.plot_obs_2D(rxLoc, d=pred_MAI, fig=fig, ax=ax3,
                               title='Predicted Amplitude (noScale)')
ax3.set_xlabel('X (m)')
ax3.set_ylabel('Y (m)')


fig = plt.figure(figsize=(5, 2.5))
ax2 = plt.subplot()
PF.Magnetics.plotModelSections(mesh, mrec_MAI, normal='y', ind=midy, subFact=2, scale=0.25, xlim=[-xlim, xlim], ylim=[-xlim, 5],
                      axs=ax2, vmin=0, vmax=vmax, contours = contours)
for midx in locx:
    ax2.add_patch(Rectangle((mesh.vectorCCx[midx-nX]-dx/2.,mesh.vectorCCz[midz-nX]-dx/2.),(2*nX+1)*dx,(2*nX+1)*dx, facecolor = 'none', edgecolor='k'))
ax2.grid(color='w', linestyle='--', linewidth=0.5)
loc = ax2.get_position()
ax2.set_position([loc.x0+0.025, loc.y0+0.025, loc.width, loc.height])
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Depth (m)')



fig = plt.figure(figsize=(5, 5))
ax3 = plt.subplot()
# vmax = mrec_MAIS.max()
vmax= None#mrec_MAIS.max()
# PF.Magnetics.plotModelSections(mesh, mrec_MAIS, normal='z', ind=-3, subFact=2, scale=0.25, xlim=[-xlim, xlim], ylim=[-xlim, xlim],
#                       title="Esus Model", axs=ax3, vmin=0, vmax=vmax, contours = contours)
# ax3.xaxis.set_visible(False)
out = PF.Magnetics.plot_obs_2D(rxLoc, d=pred_MAIS, fig=fig, ax=ax3,
                               title='Predicted Amplitude')
ax3.set_xlabel('X (m)')
ax3.set_ylabel('Y (m)')

fig = plt.figure(figsize=(5, 2.5))
ax2 = plt.subplot()
PF.Magnetics.plotModelSections(mesh, mrec_MAIS, normal='y', ind=midy, subFact=2, scale=0.25, xlim=[-xlim, xlim], ylim=[-xlim, 5],
                      axs=ax2, vmin=0, vmax=vmax, contours = contours)
for midx in locx:
    ax2.add_patch(Rectangle((mesh.vectorCCx[midx-nX]-dx/2.,mesh.vectorCCz[midz-nX]-dx/2.),(2*nX+1)*dx,(2*nX+1)*dx, facecolor = 'none', edgecolor='k'))
ax2.grid(color='w', linestyle='--', linewidth=0.5)
loc = ax2.get_position()
ax2.set_position([loc.x0+0.025, loc.y0+0.025, loc.width, loc.height])
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Depth (m)')
plt.show()

# # Plot the data
# fig = plt.figure(figsize=(6, 6))
# ax1 = plt.subplot(311)
# ax2 = plt.subplot(312)
# ax3 = plt.subplot(313)
# out = PF.Magnetics.plot_obs_2D(rxLoc, d=d_TMI, fig=fig, ax=ax1,
#                                title='TMI Data')
# out = PF.Magnetics.plot_obs_2D(rxLoc, d=d_amp, fig=fig, ax=ax2,
#                                title='Amplitude Data')
# out = PF.Magnetics.plot_obs_2D(rxLoc, d=invProb.dpred, fig=fig, ax=ax3,
#                                title='Amplitude Data')

# plt.show()
