#%%
from SimPEG import Mesh, Directives, Maps, InvProblem, Optimization, Utils
from SimPEG import DataMisfit, Inversion, Regularization, mkvc
import SimPEG.PF as PF
from SimPEG.Utils import sdiag, speye, kron3
import pylab as plt
import os
import numpy as np
from matplotlib.patches import Rectangle

# work_dir = 'C:\\Users\\DominiqueFournier\\Dropbox\\Projects\\Synthetic\\SingleBlock\\GRAV\\'
# work_dir = 'C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Kevitsa\\Modeling\\GRAV\\'
# work_dir = ".\\"
work_dir = 'C:\\Users\\DominiqueFournier\\Dropbox\\Projects\\Synthetic\\SingleBlock\\GRAV\\'
#
inpFile = 'SimPEG_GRAV.inp'
out_dir = "SimPEG_Grav_TileInv\\"
dsep = os.path.sep

padLen = 2000
maxRAM = 2
n_cpu = 12

octreeObs = [5, 10, 5, 5, 3]  # Octree levels below observation points
octreeTopo = [0, 0, 2]

meshType = 'TREE'

os.system('mkdir ' + work_dir+out_dir)


#%% User input
# Plotting parameter
#%%
# Read input file
#[mshfile, obsfile, topofile, mstart, mref, wgtfile, chi, alphas, bounds, lpnorms] = PF.Gravity.read_GRAVinv_inp(work_dir + dsep + inpFile)
driver = PF.GravityDriver.GravityDriver_Inv(work_dir + dsep + inpFile)
meshInput = driver.mesh
survey = driver.survey
topo = driver.topo
rxLoc = survey.srcField.rxList[0].locs

h = np.r_[meshInput.hx.min(), meshInput.hy.min(), meshInput.hz.min()]

# LOOP THROUGH TILES
surveyMask = np.ones(survey.nD, dtype='bool')
# Going through all problems:
# 1- Pair the survey and problem
# 2- Add up sensitivity weights
# 3- Add to the ComboMisfit

# Create first mesh outside the parallel process
padDist = np.r_[np.c_[padLen, padLen], np.c_[padLen, padLen], np.c_[padLen, 0]]

if meshType != meshInput._meshType:
    print("Creating Global Octree")
    mesh = Utils.modelutils.meshBuilder(
            rxLoc, h, padDist, meshType='TREE', meshGlobal=meshInput,
            verticalAlignment='center'
        )

    if topo is not None:
        mesh = Utils.modelutils.refineTree(
            mesh, topo, dtype='surface',
            nCpad=octreeTopo, finalize=False
        )

    mesh = Utils.modelutils.refineTree(
        mesh, rxLoc, dtype='surface',
        nCpad=octreeObs, finalize=True
    )

    if topo is not None:
        actv = Utils.surface2ind_topo(mesh, topo)
    else:
        actv = np.zeros(mesh.nC, dtype='bool')
        print(meshInput.vectorNz[-1])
        actv[mesh.gridCC[:, 2] < meshInput.vectorNz[-1]] = True

    m0 = np.ones(actv.sum()) * 1e-4

    print("Writing global Octree to file" + work_dir + out_dir + 'OctreeMeshGlobal.msh')
    Mesh.TreeMesh.writeUBC(
          mesh, work_dir + out_dir + 'OctreeMeshGlobal.msh',
          models={work_dir + out_dir + 'ActiveGlobal.act': actv}
        )


else:
    mesh = meshInput
    actv = np.zeros(mesh.nC, dtype='bool')
    actv[driver.activeCells] = True
    m0 = driver.m0  # Starting model


wrGlobal = np.zeros(int(actv.sum()))

# Loop over different tile size and break problem until memory usage is preserved
usedRAM = np.inf
count = 0
while usedRAM > maxRAM:
    print("Tiling:" + str(count))
    count += 1
    tiles = Utils.modelutils.tileSurveyPoints(rxLoc, np.ceil(rxLoc.shape[0]/count))

    X1, Y1 = tiles[0][:, 0], tiles[0][:, 1]
    X2, Y2 = tiles[1][:, 0], tiles[1][:, 1]

    ind_t = np.all([rxLoc[:, 0] >= tiles[0][0, 0], rxLoc[:, 0] <= tiles[1][0, 0],
                    rxLoc[:, 1] >= tiles[0][0, 1], rxLoc[:, 1] <= tiles[1][0, 1],
                    surveyMask], axis=0)

    meshLocal = Utils.modelutils.meshBuilder(
        rxLoc, h, padDist, meshType='TREE', meshGlobal=meshInput,
        verticalAlignment='center'
    )

    if topo is not None:
        meshLocal = Utils.modelutils.refineTree(
            meshLocal, topo, dtype='surface',
            nCpad=octreeTopo, finalize=False
        )

    meshLocal = Utils.modelutils.refineTree(
        meshLocal, rxLoc[ind_t, :], dtype='surface',
        nCpad=octreeObs, finalize=True
    )

    nD, nC = ind_t.sum()*1., meshLocal.nC*1.

    nChunks = n_cpu # Number of chunks
    cSa, cSb = int(nD/nChunks), int(nC/nChunks) # Chunk sizes
    usedRAM = nD * nC * 8. * 1e-9

    print(nD, nC, usedRAM)

# Plot data and tiles
fig, ax1 = plt.figure(), plt.subplot()
Utils.PlotUtils.plot2Ddata(rxLoc, survey.dobs, ax=ax1)
for ii in range(X1.shape[0]):
    ax1.add_patch(Rectangle((X1[ii], Y1[ii]),
                            X2[ii]-X1[ii],
                            Y2[ii]-Y1[ii],
                            facecolor='none', edgecolor='k'))
ax1.set_xlim([X1.min()-20, X2.max()+20])
ax1.set_ylim([Y1.min()-20, Y2.max()+20])
ax1.set_aspect('equal')
plt.show()

def createLocalProb(rxLoc, wrGlobal, lims, ind):

    # Grab the data for current tile
    ind_t = np.all([rxLoc[:, 0] >= lims[0], rxLoc[:, 0] <= lims[1],
                    rxLoc[:, 1] >= lims[2], rxLoc[:, 1] <= lims[3],
                    surveyMask], axis=0)

    # Remember selected data in case of tile overlap
    surveyMask[ind_t] = False

    # Create new survey
    rxLoc_t = PF.BaseGrav.RxObs(rxLoc[ind_t, :])
    srcField = PF.BaseGrav.SrcField([rxLoc_t])
    survey_t = PF.BaseGrav.LinearSurvey(srcField)
    survey_t.dobs = survey.dobs[ind_t]
    survey_t.std = survey.std[ind_t]
    survey_t.ind = ind_t

    meshLocal = Utils.modelutils.meshBuilder(
        rxLoc, h, padDist, meshType='TREE', meshGlobal=meshInput,
        verticalAlignment='center'
    )

    if topo is not None:
        meshLocal = Utils.modelutils.refineTree(
            meshLocal, topo, dtype='surface',
            nCpad=octreeTopo, finalize=False
        )

    # Refine the mesh around loc
    meshLocal = Utils.modelutils.refineTree(
        meshLocal, rxLoc[ind_t, :], dtype='surface',
        nCpad=octreeObs, finalize=True
    )

    actv_t = np.ones(meshLocal.nC, dtype='bool')

    Mesh.TreeMesh.writeUBC(
          meshLocal, work_dir + out_dir + 'OctreeMesh' + str(tt) + '.msh',
          models={work_dir + out_dir + 'Active' + str(tt) + '.act': actv_t}
        )

    print(meshLocal.nC)
    # Create reduced identity map
    tileMap = Maps.Tile((mesh, actv), (meshLocal, actv_t))

    # Create the forward model operator
    prob =  PF.Gravity.GravityIntegral(
        meshLocal, rhoMap=tileMap, actInd=actv_t, parallelized=True,
        Jpath=work_dir + out_dir + "Tile" + str(ind) + ".zarr")

    survey_t.pair(prob)

    # Data misfit function
    dmis = DataMisfit.l2_DataMisfit(survey_t)
    dmis.W = 1./survey_t.std

    wrGlobal += prob.getJtJdiag(np.ones(tileMap.P.shape[1]), W=dmis.W)

    del meshLocal

    # Create combo misfit function
    return dmis, wrGlobal


for tt in range(X1.shape[0]):

    print("Tile " + str(tt+1) + " of " + str(X1.shape[0]))

    dmis, wrGlobal = createLocalProb(rxLoc, wrGlobal, np.r_[X1[tt], X2[tt], Y1[tt], Y2[tt]], tt)

    if tt == 0:
        ComboMisfit = dmis

    else:
        ComboMisfit += dmis

# Load the rotation parameters and create gradients
mx = mesh.readModelUBC(work_dir + 'NormalX.dat')
my = mesh.readModelUBC(work_dir + 'NormalY.dat')
mz = mesh.readModelUBC(work_dir + 'NormalZ.dat')

vec = np.c_[mx, my, mz]

nC = mesh.nC
atp = Utils.matutils.xyz2atp(vec)

theta = atp[nC:2*nC]
phi = atp[2*nC:]

# theta = np.ones(mesh.nC) * 45
# phi = np.ones(mesh.nC) * 0

indActive = np.zeros(mesh.nC, dtype=bool)
indActive[actv] = True

print("Building operators")
Pac = Utils.speye(mesh.nC)[:, indActive]

Dx1 = Regularization.getDiffOpRot(mesh, np.deg2rad(0.), theta, phi, 'X')
Dy1 = Regularization.getDiffOpRot(mesh, np.deg2rad(0.), theta, phi, 'Y')
Dz1 = Regularization.getDiffOpRot(mesh, np.deg2rad(0.), theta, phi, 'Z')

Dx1 = Pac.T * Dx1 * Pac
Dy1 = Pac.T * Dy1 * Pac
Dz1 = Pac.T * Dz1 * Pac


Dx2 = Regularization.getDiffOpRot(mesh, np.deg2rad(0.), theta, phi, 'X', forward=False)
Dy2 = Regularization.getDiffOpRot(mesh, np.deg2rad(0.), theta, phi, 'Y', forward=False)
Dz2 = Regularization.getDiffOpRot(mesh, np.deg2rad(0.), theta, phi, 'Z', forward=False)

Dx2 = Pac.T * Dx2 * Pac
Dy2 = Pac.T * Dy2 * Pac
Dz2 = Pac.T * Dz2 * Pac

# Check if global mesh has regions untouched by local problem
actvGlobal = wrGlobal != 0
if actvGlobal.sum() < actv.sum():

    for ind, dmis in enumerate(ComboMisfit.objfcts):
        dmis.prob.chiMap.index = actvGlobal

wrGlobal = wrGlobal[actvGlobal]**0.5
wrGlobal = (wrGlobal/np.max(wrGlobal))

#%% Create a regularization
actvMap = Maps.InjectActiveCells(mesh, actv, 0)
actv = np.all([actv, actvMap*actvGlobal], axis=0)
actvMap = Maps.InjectActiveCells(mesh, actv, 0)
idenMap = Maps.IdentityMap(nP=int(np.sum(actv)))
# reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
# reg.norms = np.c_[driver.lpnorms].T

# if driver.eps is not None:
#     reg.eps_p = driver.eps[0]
#     reg.eps_q = driver.eps[1]

# reg.cell_weights = wrGlobal
# reg.mref = np.zeros(mesh.nC)[actv]

nC = actv.sum()

reg1 = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap,
                            gradientType='component')
reg1.norms = np.c_[driver.lpnorms].T
# reg1.alpha_x = 1#((Dx1.max() - Dx1.min())/2)**-2.
# reg1.alpha_y = 3#((Dy1.max() - Dy1.min())/2)**-2.
# reg1.alpha_z = 4
# reg1.alpha_y = 4
# reg1.eps_p = 1e-3
# reg1.eps_q = 1e-3
reg1.cell_weights = wrGlobal
reg1.mref = np.zeros(mesh.nC)[actv]

reg1.objfcts[1].regmesh._cellDiffxStencil = Dx1
reg1.objfcts[1].regmesh._aveCC2Fx = speye(nC)

reg1.objfcts[2].regmesh._cellDiffyStencil = Dy1
reg1.objfcts[2].regmesh._aveCC2Fy = speye(nC)

reg1.objfcts[3].regmesh._cellDiffzStencil = Dz1
reg1.objfcts[3].regmesh._aveCC2Fz = speye(nC)

reg2 = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap,
                            gradientType='component')
reg2.norms = np.c_[driver.lpnorms].T
# reg1.alpha_x = 1#((Dx1.max() - Dx1.min())/2)**-2.
# reg1.alpha_y = 3#((Dy1.max() - Dy1.min())/2)**-2.
# reg2.alpha_z = 4
# reg2.alpha_y = 4
# reg2.eps_p = 1e-3
reg2.cell_weights = wrGlobal
reg2.mref = np.zeros(mesh.nC)[actv]


reg2.objfcts[1].regmesh._cellDiffxStencil = Dx2
reg2.objfcts[1].regmesh._aveCC2Fx = speye(nC)

reg2.objfcts[2].regmesh._cellDiffyStencil = Dy2
reg2.objfcts[2].regmesh._aveCC2Fy = speye(nC)

reg2.objfcts[3].regmesh._cellDiffzStencil = Dz2
reg2.objfcts[3].regmesh._aveCC2Fz = speye(nC)

reg= reg1 + reg2

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=40, lower=-10, upper=10.,
                                 maxIterLS=20, maxIterCG=20, tolCG=1e-4)
invProb = InvProblem.BaseInvProblem(ComboMisfit, reg, opt)
betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e0)

# Here is where the norms are applied
# Use pick a treshold parameter empirically based on the distribution of
#  model parameters
IRLS = Directives.Update_IRLS(f_min_change=1e-3, minGNiter=1,
                              maxIRLSiter=20)

IRLS.target = driver.survey.nD
update_Jacobi = Directives.UpdatePreconditioner()
saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap, fileName=work_dir + out_dir + dsep + "GRAV_Tile")
inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, saveModel,
                                             IRLS, update_Jacobi])

# Run the inversion
mrec = inv.run(m0)


# Outputs
if mesh._meshType == 'TENSOR':
    Mesh.TensorMesh.writeUBC(mesh, work_dir + out_dir + "GRAV_Tile.msh")
    Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + "GRAV_Tile_lp.sus",
                                  actvMap*invProb.model)
    Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + "GRAV_Tile_l2.sus",
                                  actvMap*invProb.l2model)
else:

    if invProb.l2model is not None:
        Mesh.TreeMesh.writeUBC(mesh, work_dir + out_dir + 'GRAV_Tile.msh',
                               models={work_dir + out_dir + 'Model_l2.sus': actvMap*invProb.l2model})
    Mesh.TreeMesh.writeUBC(mesh, work_dir + out_dir + 'GRAV_Tile.msh',
                           models={work_dir + out_dir + 'Model_lp.sus': actvMap*mrec})


# Get predicted data for each tile and write full predicted to file
if getattr(ComboMisfit, 'objfcts', None) is not None:
    dpred = np.zeros(survey.nD)
    for ind, dmis in enumerate(ComboMisfit.objfcts):
        dpred[dmis.survey.ind] += dmis.survey.dpred(mrec)
else:
    dpred = ComboMisfit.survey.dpred(mrec)
    # PF.Magnetics.writeUBCobs(work_dir+out_dir + "Tile" + str(ind) + ".pre",
    #                          survey, survey.dpred(mrec))

Utils.io_utils.writeUBCgravityObservations(
    work_dir+out_dir + "Predicted_lp.pre", survey, dpred
    )
# Utils.io_utils.writeUBCgravityObservations(
#     work_dir + out_dir + 'Predicted_l2.pre', survey,
#     survey.dpred(invProb.l2model)
#     )

# ## OLD
# tiles = range(X1.shape[0])
# tilestep = 3
# for tt in tiles[:tilestep]:

#     midX = np.mean([X1[tt], X2[tt]])
#     midY = np.mean([Y1[tt], Y2[tt]])

#     # Create new mesh
#     padx = np.r_[dx[0]*expf**(np.asarray(range(npadxy))+1)]
#     pady = np.r_[dx[1]*expf**(np.asarray(range(npadxy))+1)]

#     hx = np.r_[padx[::-1], core_x, padx]
#     hy = np.r_[pady[::-1], core_y, pady]
# #        hz = np.r_[padb*2.,[33,26],np.ones(25)*22,[18,15,12,10,8,7,6], np.ones(18)*5,5*expf**(np.asarray(range(2*npad)))]
#     hz = mesh.hz

#     mesh_t = Mesh.TensorMesh([hx, hy, hz], 'CC0')

# #    mtemp._x0 = [x0[ii]-np.sum(padb), y0[ii]-np.sum(padb), mesh.x0[2]]

#     mesh_t._x0 = (mesh_t.x0[0] + midX, mesh_t.x0[1]+midY, mesh.x0[2])

#     Mesh.TensorMesh.writeUBC(mesh_t, work_dir + out_dir + tile_dirl2 + "MVI_S_Tile" + str(tt) + ".msh")
#     Mesh.TensorMesh.writeUBC(mesh_t, work_dir + out_dir + tile_dirlp + "MVI_S_Tile" + str(tt) + ".msh")

# #        meshes.append(mtemp)
#     # Grab the right data
#     xlim = [mesh_t.vectorCCx[npadxy], mesh_t.vectorCCx[-npadxy]]
#     ylim = [mesh_t.vectorCCy[npadxy], mesh_t.vectorCCy[-npadxy]]

#     ind_t = np.all([rxLoc[:, 0] > xlim[0], rxLoc[:, 0] < xlim[1],
#                     rxLoc[:, 1] > ylim[0], rxLoc[:, 1] < ylim[1]], axis=0)

#     if np.sum(ind_t) < 20:
#         continue

#     rxLoc_t = PF.BaseGrav.RxObs(rxLoc[ind_t, :])
#     srcField = PF.BaseGrav.SrcField([rxLoc_t])
#     survey_t = PF.BaseGrav.LinearSurvey(srcField)
#     survey_t.dobs = survey.dobs[ind_t]
#     survey_t.std = survey.std[ind_t]

#     # Extract model from global to local mesh
#     if driver.topofile is not None:
#         topo = np.genfromtxt(work_dir + driver.topofile,
#                              skip_header=1)
#         actv = Utils.surface2ind_topo(mesh_t, topo, 'N')
#         actv = np.asarray(np.where(mkvc(actv))[0], dtype=int)
#     else:
#         actv = np.ones(mesh_t.nC, dtype='bool')

#     nC = len(actv)
#     print("Tile "+str(tt))
#     print(nC, np.sum(ind_t))
#     # Create active map to go from reduce space to full
#     actvMap = Maps.InjectActiveCells(mesh_t, actv, 0)

#     # Create identity map
#     idenMap = Maps.IdentityMap(nP=3*nC)

#     mstart = np.ones(3*len(actv))*1e-4
#     #%% Run inversion
#     prob = PF.Gravity.GravityIntegral(mesh_t, rhoMap=idenMap, actInd=actv)
#     prob.solverOpts['accuracyTol'] = 1e-4

#     survey.pair(prob)

#     # Write out the predicted file and generate the forward operator
#     pred = prob.fields(mstart)

#     PF.Gravity.writeUBCobs(work_dir + dsep + 'Pred0.dat', survey, pred)

#     wr = np.sum(prob.F**2., axis=0)**0.5
#     wr = (wr/np.max(wr))

#     #%% Create inversion objects
#     reg = Regularization.Sparse(mesh, indActive=actv, mapping=staticCells)
#     reg.mref = driver.mref[dynamic]
#     reg.cell_weights = wr
#     reg.norms = driver.lpnorms
#     if driver.eps is not None:
#         reg_a.eps_p = driver.eps[0]
#         reg_a.eps_q = driver.eps[1]

#     opt = Optimization.ProjectedGNCG(maxIter=100, lower=driver.bounds[0],upper=driver.bounds[1], maxIterLS = 20, maxIterCG= 10, tolCG = 1e-3)
#     dmis = DataMisfit.l2_DataMisfit(survey)
#     dmis.W = 1./wd
#     invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

#     betaest = Directives.BetaEstimate_ByEig()
#     IRLS = Directives.Update_IRLS(f_min_change=1e-4, minGNiter=3)
#     update_Jacobi = Directives.UpdatePreCond()
#     saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap)
#     saveModel.fileName = work_dir + out_dir + 'SimPEG_GRAV'
#     inv = Inversion.BaseInversion(invProb, directiveList=[betaest, IRLS,
#                                                           update_Jacobi,
#                                                           saveModel])

#     # Run inversion
#     mrec = inv.run(mstart)

#     # Plot predicted
#     pred = prob.fields(mrec)
#     #PF.Magnetics.plot_obs_2D(rxLoc,pred,wd,'Predicted Data')
#     #PF.Magnetics.plot_obs_2D(rxLoc,(d-pred),wd,'Residual Data')
#     survey.dobs = pred
#     # PF.Gravity.plot_obs_2D(survey, 'Observed Data')
#     print("Final misfit:" + str(np.sum(((d-pred)/wd)**2.)))

#     #%% Plot out a section of the model

#     yslice = midx

#     m_out = actvMap*staticCells*invProb.l2model

#     # Write result
#     Mesh.TensorMesh.writeModelUBC(mesh, work_dir + dsep + 'SimPEG_inv_l2l2.den', m_out)

#     m_out = actvMap*staticCells*mrec
#     # Write result
#     Mesh.TensorMesh.writeModelUBC(mesh, work_dir + dsep + 'SimPEG_inv_lplq.den', m_out)


# Nan aircells for plotting
# m_out[m_out==-100] = np.nan

# plt.figure()
# ax = plt.subplot(221)
# mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-10, clim = (mrec.min(), mrec.max()))
# plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
# plt.title('Z: ' + str(mesh.vectorCCz[-5]) + ' m')
# plt.xlabel('x');plt.ylabel('z')
# plt.gca().set_aspect('equal', adjustable='box')

# ax = plt.subplot(222)
# mesh.plotSlice(m_out, ax = ax, normal = 'Z', ind=-15, clim = (mrec.min(), mrec.max()))
# plt.plot(np.array([mesh.vectorCCx[0],mesh.vectorCCx[-1]]), np.array([mesh.vectorCCy[yslice],mesh.vectorCCy[yslice]]),c='w',linestyle = '--')
# plt.title('Z: ' + str(mesh.vectorCCz[-15]) + ' m')
# plt.xlabel('x');plt.ylabel('z')
# plt.gca().set_aspect('equal', adjustable='box')


# ax = plt.subplot(212)
# mesh.plotSlice(m_out, ax = ax, normal = 'Y', ind=yslice, clim = (mrec.min(), mrec.max()))
# plt.title('Cross Section')
# plt.xlabel('x');plt.ylabel('z')
# plt.gca().set_aspect('equal', adjustable='box')

# plt.figure()
# ax = plt.subplot(121)
# plt.hist(reg.l2model,100)
# plt.yscale('log', nonposy='clip')
# plt.title('Histogram of model values - Smooth')
# ax = plt.subplot(122)
# plt.hist(reg.regmesh.cellDiffxStencil*(staticCells*reg.l2model),100)
# plt.yscale('log', nonposy='clip')
# plt.title('Histogram of model gradient values - Smooth')

# #%% Plot out a section of the model

# yslice = midx

