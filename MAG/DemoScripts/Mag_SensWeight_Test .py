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
B = [50000, 90, 0]

# Create a mesh
dx = 5.
npad = 10
hxind = [(dx, npad, -1.3), (dx, 24), (dx, npad, 1.3)]
hyind = [(dx, npad, -1.3), (dx, 24), (dx, npad, 1.3)]
hzind = [(dx, npad, -1.3), (dx, 20)]

mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CC0')
mesh.x0[2] -= mesh.vectorNz[-1]
norms = [2, 2, 2, 2]
susc = 0.025
nX = 2

# Get index of the center of block
locx = [int(mesh.nCx/2)]#[int(mesh.nCx/2)-3, int(mesh.nCx/2)+3]
midy = int(mesh.nCy/2)
midz = -6

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
xr = np.linspace(-50., 50., 10)
yr = np.linspace(-50., 50., 10)
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
M = PF.Magnetics.dipazm_2_xyz(np.ones(nC) * 90., np.ones(nC) * 0.)
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
d_TMI = prob.Intrgl_Fwr_Op(m=model, recType='tmi')

ndata = survey.nD

# Add noise and uncertainties
# We add some random Gaussian noise (1nT)
floor = d_TMI.max()*0.05
d_TMI += np.random.randn(len(d_TMI))*floor
wd = np.ones(len(d_TMI))*floor  # Assign flat uncertainties
survey.dobs = d_TMI
survey.std = wd

rxLoc = survey.srcField.rxList[0].locs

# For comparison, let's run the inversion assuming an induced response
M = PF.Magnetics.dipazm_2_xyz(np.ones(nC) * B[1], np.ones(nC) * B[2])

# Reset the magnetization
prob.M = M
prob.forwardOnly = False
prob._F = None

fig = plt.figure(figsize=(12, 6))

exp = np.linspace(2, -8, 6)
eps = 10**exp

eps[-1] = 0.
for ii in range(6):

    # Create a regularization function, in this case l2l2
    wr = (np.sum(prob.F**2. + eps[ii]**2., axis=0))**0.5
    # wr = (wr/np.max(wr))


    # Create a regularization
    reg_Susc = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
    reg_Susc.cell_weights = wr
    reg_Susc.norms = [2, 2, 2, 2]
    reg_Susc.alpha_x = 0
    reg_Susc.alpha_y = 0
    reg_Susc.alpha_z = 0
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


    #%% Plot models
    from matplotlib.patches import Rectangle
    ax2 = plt.subplot(2, 3, ii+1)
    contours = [0.01]
    xlim = 150.

    vmax = mrec_sus.max()
    vmin = -0.001
    # PF.Magnetics.plotModelSections(mesh, mrec_sus, normal='z', ind=-3, subFact=2, scale=0.25, xlim=[-xlim, xlim], ylim=[-xlim, xlim],
    #                       title="Esus Model", axs=ax3, vmin=0, vmax=vmax, contours = contours)
    # ax3.xaxis.set_visible(False)

    # out = PF.Magnetics.plot_obs_2D(rxLoc, d=pred_sus, fig=fig, ax=ax3,
    #                                )
    # ax3.set_xlabel('X (m)')
    # ax3.set_ylabel('Y (m)')

    # fig = plt.figure(figsize=(5, 2.5))
    # ax2 = plt.subplot()
    # PF.Magnetics.plotModelSections(mesh, model, normal='y', ind=midy, subFact=2, scale=0.25, xlim=[-xlim, xlim], ylim=[-xlim, 5],
    #                       axs=ax2, vmin=vmin, vmax=vmax, contours = contours)
    # for midx in locx:
    #     ax2.add_patch(Rectangle((mesh.vectorCCx[midx-nX]-dx/2.,mesh.vectorCCz[midz-nX]-dx/2.),(2*nX+1)*dx,(2*nX+1)*dx, facecolor = 'none', edgecolor='k'))
    # ax2.grid(color='w', linestyle='--', linewidth=0.5)
    # loc = ax2.get_position()
    # ax2.set_position([loc.x0+0.025, loc.y0+0.025, loc.width, loc.height])
    # ax2.set_xlabel('X (m)')
    # ax2.set_ylabel('Depth (m)')

    ax2, im2, cbar = PF.Magnetics.plotModelSections(mesh, mrec_sus, normal='y', ind=midy, subFact=2, scale=0.25, xlim=[-xlim, xlim], ylim=[-xlim, 5],
                          axs=ax2, vmin=vmin, vmax=vmax, contours = contours)
    cbar.remove()
    for midx in locx:
        ax2.add_patch(Rectangle((mesh.vectorCCx[midx-nX]-dx/2.,mesh.vectorCCz[midz-nX]-dx/2.),(2*nX+1)*dx,(2*nX+1)*dx, facecolor = 'none', edgecolor='k'))
    ax2.grid(color='w', linestyle='--', linewidth=0.5)
    loc = ax2.get_position()
    ax2.set_position([loc.x0, loc.y0, loc.width*1.2, loc.height*1.2])
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Depth (m)')

    if ii < 5:
      ax2.set_title('$\epsilon=\;10^{' + str(exp[ii]) + '}$')
    else:
      ax2.set_title('$\epsilon=0$')


plt.show()
