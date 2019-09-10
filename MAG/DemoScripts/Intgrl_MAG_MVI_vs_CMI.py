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
B = (50000, 90, 0)

# Create a mesh
dx = 5.

hxind = [(dx, 5, -1.3), (dx, 16), (dx, 5, 1.3)]
hyind = [(dx, 5, -1.3), (dx, 16), (dx, 5, 1.3)]
hzind = [(dx, 5, -1.3), (dx, 8)]

mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCC')
#mesh.x0[2] -= mesh.vectorNz[-1]

susc = 0.1
nX = 1

# Get index of the center of block
locx = [int(mesh.nCx/2)]
midy = int(mesh.nCy/2)
midz = -3

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
xr = np.linspace(-30., 30., 20)
yr = np.linspace(-30., 30., 20)
X, Y = np.meshgrid(xr, yr)

# Move the observation points 5m above the topo
Z = np.ones(X.shape)*mesh.vectorNz[-1] + dx

# Create a MAGsurvey
xyzLoc = np.c_[Utils.mkvc(X.T), Utils.mkvc(Y.T), Utils.mkvc(Z.T)]
rxLoc = PF.BaseMag.RxObs(xyzLoc)
srcField = PF.BaseMag.SrcField([rxLoc], param=B)
survey = PF.BaseMag.LinearSurvey(srcField)

# We can now create a susceptibility model and generate data
# Here a simple block in half-space
model = np.zeros((mesh.nCx, mesh.nCy, mesh.nCz))
for midx in locx:
    model[(midx-nX-1):(midx+nX), (midy-nX-1):(midy+nX), (midz-nX-1):(midz+nX)] = susc

model = Utils.mkvc(model)


# We create a magnetization model different than the inducing field
# to simulate remanent magnetization. Let's do something simple [45,90]
M = PF.Magnetics.dipazm_2_xyz(np.ones(nC) * 0., np.ones(nC) * 90.)

m = mkvc(sp.diags(model, 0) * M)
# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

# Create reduced identity map
idenMap = Maps.IdentityMap(nP=nC)

# Create the forward problem (forwardOnly)
prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap, actInd=actv,
                                     M=M, forwardOnly=True, rtype = 'xyz')

# Pair the survey and problem
survey.pair(prob)

# Compute forward model some data
d = prob.fields(model)

nD = survey.nD
x = d[:nD]
y = d[nD:2*nD]
z = d[2*nD:]

# Project to TMI and add noise and uncertainties
# We add some random Gaussian noise (1nT)
d_TMI = np.squeeze(prob.ProjTMI.dot(np.c_[x, y, z].T))
wd = np.ones(len(d_TMI))  # Assign flat uncertainties
survey.dobs = d_TMI
survey.std = wd

# Compute Amplitude data
d_amp = np.sqrt(x**2. + y**2. + z**2.)



# # STEP 3: RUN AMPLITUDE INVERSION ##

# Now that we have |B| data, we can invert. This is a non-linear inversion,
# which requires some special care for the sensitivity weighting
# (see Directives)

# Create active map to go from reduce space to full
nC = int(len(actv))

# Create identity map
idenMap = Maps.IdentityMap(nP=nC)

# Create the forward model operator
prob = PF.Magnetics.MagneticAmplitude(mesh, chiMap=idenMap,
                                      actInd=actv)

# Define starting model
mstart = np.ones(len(actv))*1e-4
prob.chi = mstart

# Change the survey to xyz components
survey.srcField.rxList[0].rxType = 'xyz'

# Pair the survey and problem
survey.pair(prob)

# Re-set the observations to |B|
survey.dobs = d_amp

# Create a sparse regularization
reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
reg.mref = mstart*0.

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = 2./d_amp.min()

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=100, lower=0., upper=1.,
                                 maxIterLS=20, maxIterCG=10,
                                 tolCG=1e-3)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=1e+6)

# Here is the list of directives
betaest = Directives.BetaEstimate_ByEig()

# Specify the sparse norms
IRLS = Directives.Update_IRLS(norms=([0, 1, 1, 1]),
                              eps=None, f_min_change=1e-3,
                              minGNiter=3,
                              chifact = .25)

# Special directive specific to the mag amplitude problem. The sensitivity
# weights are update between each iteration.
update_Jacobi = Directives.Amplitude_Inv_Iter()

# Put all together
inv = Inversion.BaseInversion(invProb,
                              directiveList=[IRLS, update_Jacobi])

# Invert
mrec_MAI_lp = inv.run(mstart)
mrec_MAI_l2 = reg.l2model
    
# # RUN CMI INVERSION
# Create active map to go from reduce set to full
# Creat reduced identity map
idenMap = Maps.IdentityMap(nP=3*nC)

# Create the forward model operator
prob = PF.Magnetics.MagneticVector(mesh, chiMap=idenMap,
                                     actInd=actv)

# Pair the survey and problem
survey = PF.BaseMag.LinearSurvey(srcField)
survey.srcField.rxList[0].rxType = 'tmi'

survey.dobs = d_TMI
survey.std = wd

survey.pair(prob)


# # RUN CMI with l2 # #

# Create rescaled weigths
mamp = (mrec_MAI_l2[actv]/mrec_MAI_l2[actv].max() + 1e-2)**-1.

# Update the sensitivity weights with amplitude weights added
reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap, nSpace=3)
wr = np.sum(prob.G**2., axis=0)**0.5
wr = (wr/np.max(wr))*np.r_[mamp, mamp, mamp]
reg.cell_weights = wr
reg.mref = np.zeros(3*nC)

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = 1./survey.std

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=30, lower=-10., upper=10.,
                                 maxIterCG=20, tolCG=1e-3)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
betaest = Directives.BetaEstimate_ByEig()

betaCool = Directives.BetaSchedule(coolingFactor=2., coolingRate=1)

update_Jacobi = Directives.Update_lin_PreCond()
targetMisfit = Directives.TargetMisfit()

inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, update_Jacobi, betaCool, targetMisfit])

mrec_CMI_l2 = inv.run(np.ones(3*len(actv))*1e-4)


# Create identity map

# # RUN CMI WITH LP # #
# Create rescaled weigths
mamp = (mrec_MAI_lp[actv]/mrec_MAI_lp[actv].max() + 1e-2)**-1.

# Update the sensitivity weights with amplitude weights added
prob = PF.Magnetics.MagneticVector(mesh, chiMap=idenMap,
                                     actInd=actv)

# Pair the survey and problem
survey = PF.BaseMag.LinearSurvey(srcField)
survey.srcField.rxList[0].rxType = 'tmi'

survey.dobs = d_TMI
survey.std = wd

survey.pair(prob)


idenMap = Maps.IdentityMap(nP=3*nC)
reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap, nSpace=3)
wr = np.sum(prob.G**2., axis=0)**0.5
wr = (wr/np.max(wr))*np.r_[mamp, mamp, mamp]
reg.cell_weights = wr
reg.mref = np.zeros(3*nC)

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.Wd = 1./survey.std

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=30, lower=-10., upper=10.,
                                 maxIterCG=20, tolCG=1e-3)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
betaest = Directives.BetaEstimate_ByEig()

betaCool = Directives.BetaSchedule(coolingFactor=2., coolingRate=1)

update_Jacobi = Directives.Update_lin_PreCond()
targetMisfit = Directives.TargetMisfit()

inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, update_Jacobi, betaCool, targetMisfit])

mrec_CMI_lp = inv.run(np.ones(3*len(actv))*1e-4)



#m_l2 = actvMap * reg.l2model[0:nC]
#m_l2[m_l2==-100] = np.nan


sub = 2
#%% Plot the result
from matplotlib.patches import Rectangle

fig = plt.figure(figsize=(16,7))
ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

ypanel = int(mesh.nCy/2)-1
zpanel = -4

vmin = 0.
xlim = [-50, 50]

vmax = mrec_MAI_l2.max()
PF.Magnetics.plotModelSections(mesh, mrec_MAI_l2, normal='y',
                               ind=ypanel, axs=ax1,
                               xlim=xlim,
                               ylim=(mesh.vectorNz[3], mesh.vectorNz[-1]+dx),
                               vmin=vmin, vmax=vmax)

for midx in locx:
    ax1.add_patch(Rectangle((mesh.vectorCCx[midx-nX-1]-dx/2.,mesh.vectorCCz[midz-nX-1]-dx/2.),3*dx,3*dx, facecolor = 'none', edgecolor='w'))

ax1.xaxis.set_visible(False)


vmax = mrec_CMI_l2.max()
scale = mrec_CMI_l2.max()/m.max()*0.75
PF.Magnetics.plotModelSections(mesh, mrec_CMI_l2, normal='y',
                               ind=ypanel, axs=ax2,
                               xlim=xlim, scale = scale, vec ='w',
                               ylim=(mesh.vectorNz[3], mesh.vectorNz[-1]+dx),
                               vmin=vmin, vmax=vmax)

for midx in locx:
    ax2.add_patch(Rectangle((mesh.vectorCCx[midx-nX-1]-dx/2.,mesh.vectorCCz[midz-nX-1]-dx/2.),3*dx,3*dx, facecolor = 'none', edgecolor='w'))

ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)

vmax = mrec_MAI_lp.max()
PF.Magnetics.plotModelSections(mesh, mrec_MAI_lp, normal='y',
                               ind=ypanel, axs=ax3,
                               xlim=xlim,
                               ylim=(mesh.vectorNz[3], mesh.vectorNz[-1]+dx),
                               vmin=vmin, vmax=vmax)

for midx in locx:
    ax3.add_patch(Rectangle((mesh.vectorCCx[midx-nX-1]-dx/2.,mesh.vectorCCz[midz-nX-1]-dx/2.),3*dx,3*dx, facecolor = 'none', edgecolor='w'))



vmax = mrec_CMI_lp.max()
scale = mrec_CMI_lp.max()/m.max()*0.75
ax4, im4, cbar = PF.Magnetics.plotModelSections(mesh, mrec_CMI_lp, normal='y',
                               ind=ypanel, axs=ax4,
                               xlim=xlim, scale=scale, vec = 'w',
                               ylim=(mesh.vectorNz[3], mesh.vectorNz[-1]+dx),
                               vmin=vmin, vmax=vmax)
for midx in locx:
    ax4.add_patch(Rectangle((mesh.vectorCCx[midx-nX-1]-dx/2.,mesh.vectorCCz[midz-nX-1]-dx/2.),3*dx,3*dx, facecolor = 'none', edgecolor='w'))

ax4.yaxis.set_visible(False)