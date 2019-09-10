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
from SimPEG import Mesh, Directives, Maps, InvProblem, Optimization
from SimPEG import DataMisfit, Inversion, Utils, Regularization
from SimPEG.Utils import mkvc
import SimPEG.PF as PF
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree
import os

work_dir = '/tera_raid/dfournier/Kevitsa/MAG/Aiborne/'
out_dir = "SimPEG_MVI_S_TileInv\\"
input_file = "SimPEG_MAG.inp"
padLen = 5000
expf = 1.3
dwnFact = 0.25
# %%
# Read in the input file which included all parameters at once
# (mesh, topo, model, survey, inv param, etc.)
driver = PF.MagneticsDriver.MagneticsDriver_Inv(work_dir + input_file)

os.system('if not exist ' + work_dir + out_dir + ' mkdir ' + work_dir+out_dir)

# Access the mesh and survey information
mesh = driver.mesh
survey = driver.survey
actv = np.zeros(mesh.nC, dtype='bool')
actv[driver.activeCells] = True

nD = int(survey.nD*dwnFact)
print("nD ratio:" + str(nD) +'\\' + str(survey.nD) )
indx = np.unique(np.random.randint(0, high=survey.nD, size=nD))
# Create a new downsampled survey
locXYZ = survey.srcField.rxList[0].locs[indx,:]

dobs = survey.dobs
std = survey.std

rxLoc = PF.BaseGrav.RxObs(locXYZ)
srcField = PF.BaseMag.SrcField([rxLoc], param=survey.srcField.param)
survey = PF.BaseMag.LinearSurvey(srcField)
survey.dobs = dobs[indx]
survey.std = std[indx]

rxLoc = survey.srcField.rxList[0].locs
#tree = cKDTree(np.c_[mesh.gridCC[actv, 0],
#                     mesh.gridCC[actv, 1],
#                     mesh.gridCC[actv, 2]])


# # TILE THE PROBLEM
# Define core mesh properties
h = np.r_[[np.min(np.r_[mesh.hx.min(), mesh.hy.min(), mesh.hz.min()])]*3]

maxNpoints = 500

tiles = Utils.modelutils.tileSurveyPoints(rxLoc, maxNpoints)

X1, Y1 = tiles[0][:,0], tiles[0][:, 1]
X2, Y2 = tiles[1][:,0], tiles[1][:, 1]


# LOOP THROUGH TILES
dx = [mesh.hx.min(), mesh.hy.min()]
surveyMask = np.ones(survey.nD, dtype='bool')
# Going through all problems:
# 1- Pair the survey and problem
# 2- Add up sensitivity weights
# 3- Add to the ComboMisfit
wrGlobal = np.zeros(3*int(actv.sum()))
probSize = 0
for tt in range(X1.shape[0]):

    print("Tile " + str(tt+1) + " of " + str(X1.shape[0]))
    if tt == 0:
        tree = cKDTree(np.c_[mesh.gridCC[actv, 0],
                             mesh.gridCC[actv, 1],
                             mesh.gridCC[actv, 2]])

    # Grab the data for current tile
    ind_t = np.all([rxLoc[:, 0] >= X1[tt], rxLoc[:, 0] <= X2[tt],
                    rxLoc[:, 1] >= Y1[tt], rxLoc[:, 1] <= Y2[tt],
                    surveyMask], axis=0)

    # Remember selected data in case of tile overlap
    surveyMask[ind_t] = False

    # Create new survey
    rxLoc_t = PF.BaseMag.RxObs(rxLoc[ind_t, :])
    srcField = PF.BaseMag.SrcField([rxLoc_t], param=survey.srcField.param)
    survey_t = PF.BaseMag.LinearSurvey(srcField)
    survey_t.dobs = survey.dobs[ind_t]
    survey_t.std = survey.std[ind_t]
    survey_t.ind = ind_t

    padDist = np.r_[np.c_[padLen, padLen], np.c_[padLen, padLen], np.c_[padLen, 0]]

    if tt == 0:
        meshTree = Utils.modelutils.meshBuilder(rxLoc[ind_t, :], h,
                                              padDist, meshGlobal=mesh,
                                              meshType='TREE',
                                              padCore=np.r_[3, 3, 3])
        core = meshTree.vol == meshTree.vol.min()
        center = np.percentile(meshTree.gridCC[core,:], 50,
                               axis=0, interpolation='nearest')

    mesh_t = meshTree.copy()

    tileCenter = np.r_[np.mean([X1[tt], X2[tt]]), np.mean([Y1[tt], Y2[tt]]), center[2]]

    ind = closestPoints(mesh, tileCenter, gridLoc='CC')

    shift = np.squeeze(mesh.gridCC[ind, :]) - center

    mesh_t.x0 += shift
    mesh_t.number()

    # Extract model from global to local mesh
#    if driver.topofile is not None:
#        topo = np.genfromtxt(work_dir + driver.topofile,
#                             skip_header=1)
#        actv_t = Utils.surface2ind_topo(mesh_t, topo, 'N')
#        # actv_t = np.asarray(np.where(mkvc(actv))[0], dtype=int)
#    else:
    actv_t = np.ones(mesh_t.nC, dtype='bool')

    # Create reduced identity map
    tileMap = Maps.Tile((mesh, actv), (mesh_t, actv_t))
    tileMap.nCell = 27
    tileMap.nBlock = 3
    # Create the forward model operator
    prob = PF.Magnetics.MagneticVector(mesh_t, chiMap=tileMap, actInd=actv_t)
    survey_t.pair(prob)

    # Data misfit function
    dmis = DataMisfit.l2_DataMisfit(survey_t)
    dmis.W = 1./survey_t.std
#    ncell = Utils.mkvc(np.sum(prob.chiMap.deriv(0) > 0, axis=1))
#    ncell[ncell>0] = (1./ncell[ncell>0]).copy()
#    SCALE = Utils.sdiag(mesh_t.vol**(0))
    for ii in range(prob.F.shape[0]):
        wrGlobal += ((prob.F[ii, :])*(prob.chiMap.deriv(0)))**2.

#    wrGlobal += np.abs(prob.Jtvec(0, prob.Jvec(0, np.ones(mesh.nC)*1e-4)))
#    wrGlobal += prob.chiMap.deriv(0).T*wr

    # Create combo misfit function
    if tt == 0:
        ComboMisfit = dmis

    else:
        ComboMisfit += dmis

    # Add problem size
    probSize += prob.F.shape[0] * prob.F.shape[1] * 32 / 4

print('Sum of all problems:' + str(probSize*1e-6) + ' Mb')
# Scale global weights for regularization

# Check if global mesh has regions untouched by local problem
actvGlobal = wrGlobal != 0
activeMeshGlobal = wrGlobal[:mesh.nC] != 0
if actvGlobal.sum() < actv.sum():

    for ind, dmis in enumerate(ComboMisfit.objfcts):
        dmis.prob.chiMap.index = actvGlobal

wrGlobal = wrGlobal[actvGlobal]**0.5
wrGlobal = (wrGlobal/np.max(wrGlobal))

#%% Create a regularization
actv = np.all([actv, activeMeshGlobal], axis=0)
actvMap = Maps.InjectActiveCells(mesh, actv, 0)
actvMapAmp = Maps.InjectActiveCells(mesh, actv, -100)

nC = int(np.sum(actv))
# Create a block diagonal regularization
wires = Maps.Wires(('p', nC), ('s', nC), ('t', nC))

# Create a regularization
reg_p = Regularization.Sparse(mesh, indActive=actv, mapping=wires.p)
reg_p.cell_weights = (wires.p * wrGlobal)
reg_p.norms = [2, 2, 2, 2]

reg_s = Regularization.Sparse(mesh, indActive=actv, mapping=wires.s)
reg_s.cell_weights = (wires.s * wrGlobal)
reg_s.norms = [2, 2, 2, 2]

reg_t = Regularization.Sparse(mesh, indActive=actv, mapping=wires.t)
reg_t.cell_weights = (wires.t * wrGlobal)
reg_t.norms = [2, 2, 2, 2]

reg = reg_p + reg_s + reg_t
reg.mref = np.zeros(3*nC)

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=7, lower=-10., upper=10.,
                                 maxIterCG=20, tolCG=1e-3)

invProb = InvProblem.BaseInvProblem(ComboMisfit, reg, opt)
betaest = Directives.BetaEstimate_ByEig()

# Here is where the norms are applied
IRLS = Directives.Update_IRLS(f_min_change=1e-4,
                              minGNiter=3, beta_tol=1e-2)

update_Jacobi = Directives.UpdateJacobiPrecond()
targetMisfit = Directives.TargetMisfit()

saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap)
saveModel.fileName = work_dir + out_dir + 'MVI_C'
inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, IRLS, update_Jacobi,
                                              saveModel])

m0 = np.ones(3*nC)*1e-4
mrec_MVI = inv.run(m0)

x = actvMap * (wires.p * mrec_MVI)
y = actvMap * (wires.s * mrec_MVI)
z = actvMap * (wires.t * mrec_MVI)

amp =  (np.sum(np.c_[x, y, z]**2., axis=1))**0.5
#amp[actv!=True] = -100
Mesh.TensorMesh.writeModelUBC(mesh, work_dir+out_dir + 'MVI_C_amp.sus', amp)

beta = invProb.beta

# %% RUN MVI-S WITH SPARSITY

# # STEP 3: Finish inversion with spherical formulation
mstart = PF.Magnetics.xyz2atp(mrec_MVI)
prob.coordinate_system = 'spherical'
prob.model = mstart

# Create a block diagonal regularization
wires = Maps.Wires(('amp', nC), ('theta', nC), ('phi', nC))

# Create a regularization
reg_a = Regularization.Sparse(mesh, indActive=actv, mapping=wires.amp)
reg_a.norms = driver.lpnorms[:4]
if driver.eps is not None:
    reg_a.eps_p = driver.eps[0]
    reg_a.eps_q = driver.eps[1]
else:
    reg_a.eps_p = np.percentile(np.abs(mstart[:nC]), 95)


reg_t = Regularization.Sparse(mesh, indActive=actv, mapping=wires.theta)
reg_t.alpha_s = 0.  # No reference angle
reg_t.space = 'spherical'
reg_t.norms = driver.lpnorms[4:8]
reg_t.eps_q = 1e-2
# reg_t.alpha_x, reg_t.alpha_y, reg_t.alpha_z = 0.25, 0.25, 0.25

reg_p = Regularization.Sparse(mesh, indActive=actv, mapping=wires.phi)
reg_p.alpha_s = 0.  # No reference angle
reg_p.space = 'spherical'
reg_p.norms = driver.lpnorms[8:]
reg_p.eps_q = 1e-2

reg = reg_a + reg_t + reg_p
reg.mref = np.zeros(3*nC)

Lbound = np.kron(np.asarray([0, -np.inf, -np.inf]), np.ones(nC))
Ubound = np.kron(np.asarray([10, np.inf, np.inf]), np.ones(nC))


# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=40,
                                 lower=Lbound,
                                 upper=Ubound,
                                 maxIterLS=10,
                                 maxIterCG=20, tolCG=1e-3,
                                 stepOffBoundsFact=1e-8)

invProb = InvProblem.BaseInvProblem(ComboMisfit, reg, opt, beta=beta*10)
#  betaest = Directives.BetaEstimate_ByEig()

# Here is where the norms are applied
IRLS = Directives.Update_IRLS(f_min_change=1e-4,
                              minGNiter=3, beta_tol=1e-2,
                              coolingRate=3)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=beta)

# Special directive specific to the mag amplitude problem. The sensitivity
# weights are update between each iteration.
ProjSpherical = Directives.ProjSpherical()
update_SensWeight = Directives.UpdateSensWeighting()
update_Jacobi = Directives.UpdateJacobiPrecond()
saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap)
saveModel.fileName = work_dir+out_dir + 'MVI_S'

inv = Inversion.BaseInversion(invProb,
                              directiveList=[ProjSpherical, IRLS, update_SensWeight,
                                             update_Jacobi, saveModel])

mrec_MVI_S = inv.run(mstart)

Mesh.TensorMesh.writeModelUBC(mesh, work_dir+out_dir + 'MVI_S_amp.sus',
                              actvMap * (mrec_MVI_S[:nC]))
Mesh.TensorMesh.writeModelUBC(mesh, work_dir+out_dir + 'MVI_S_theta.sus',
                              actvMap * (mrec_MVI_S[nC:2*nC]))
Mesh.TensorMesh.writeModelUBC(mesh, work_dir+out_dir + 'MVI_S_phi.sus',
                              actvMap * (mrec_MVI_S[2*nC:]))
