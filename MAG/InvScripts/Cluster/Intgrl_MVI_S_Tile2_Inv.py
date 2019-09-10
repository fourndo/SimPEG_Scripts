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
from SimPEG.Utils import mkvc
import SimPEG.PF as PF
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import gc

#work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Kevitsa\\Modeling\\MAG\\"
#work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\\Research\\Modelling\\Synthetic\\Block_Gaussian_topo\\"
# work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Modelling\\Synthetic\\Nut_Cracker\\"
#work_dir = "C:\\Egnyte\\Private\\dominiquef\\Projects\\4559_CuMtn_ZTEM\\Modeling\\MAG\\A1_Fenton\\"
#work_dir = 'C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Kevitsa\\Modeling\\MAG\\Aiborne\\'
work_dir = '/tera_raid/dfournier/Kevitsa/MAG/Aiborne/'
out_dir = "SimPEG_PF_Inv_MVI_S/"
tile_dirl2 = 'Tiles_l2/'
tile_dirlp = 'Tiles_lp/'
input_file = "SimPEG_MAG.inp"

max_mcell = 1.5e+5
min_Olap = 1e+3

# %%
# Read in the input file which included all parameters at once
# (mesh, topo, model, survey, inv param, etc.)
driver = PF.MagneticsDriver.MagneticsDriver_Inv(work_dir + input_file)

os.system('if not exist ' + work_dir + out_dir + ' mkdir ' + work_dir+out_dir)
os.system('if not exist ' + work_dir + out_dir + tile_dirlp + ' mkdir ' + work_dir+out_dir+ tile_dirlp)
os.system('if not exist ' + work_dir + out_dir + tile_dirl2 + ' mkdir ' + work_dir+out_dir+ tile_dirl2)

# Access the mesh and survey information
mesh = driver.mesh
survey = driver.survey
actv = driver.activeCells

# # TILE THE PROBLEM
lx = int(np.floor(np.sqrt(max_mcell/mesh.nCz)))
# #%% Create tiles
# # Begin tiling

# In the x-direction
ntile = 1
Olap = -1
dx = [mesh.hx.min(), mesh.hy.min()]
# Cell size


while Olap < min_Olap:

    ntile += 1

    # Set location of SW corners
    x0 = np.asarray([mesh.vectorNx[0], mesh.vectorNx[-1] - lx*dx[0]])

    dx_t = np.round((x0[1] - x0[0]) / ((ntile-1) * dx[0]))

    if ntile > 2:
        x1 = np.r_[x0[0],
                   x0[0] + np.cumsum(np.ones(ntile-2) * dx_t * dx[0]),
                   x0[1]]
    else:
        x1 = np.asarray([x0[0], x0[1]])

    x2 = x1 + lx*dx[0]

#     y1 = np.ones(x1.shape[0])*np.min(locs[:,1]);
#     y2 = np.ones(x1.shape[0])*(np.min(locs[:,1]) + nCx*dx);

    Olap = x1[0] + lx*dx[0] - x1[1]

# Save x-corner location
xtile = np.c_[x1, x2]

# In the Y-direction
ntile = 1
Olap = -1

# Cell size

while Olap < min_Olap:

    ntile += 1

    # Set location of SW corners
    y0 = np.asarray([mesh.vectorNy[0], mesh.vectorNy[-1] - lx*dx[0]])

    dy_t = np.round((y0[1] - y0[0]) / ((ntile-1) * dx[0]))

    if ntile > 2:
        y1 = np.r_[y0[0],
                   y0[0] + np.cumsum(np.ones(ntile-2) * dy_t * dx[0]),
                   y0[1]]
    else:
        y1 = np.asarray([y0[0], y0[1]])

    y2 = y1 + lx*dx[0]

#     x1 = np.ones(y1.shape[0])*np.min(locs[:,0]);
#     x2 = np.ones(y1.shape[0])*(np.min(locs[:,0]) + nCx*dx);

    Olap = y1[0] + lx*dx[0] - y1[1]

# Save x-corner location
ytile = np.c_[y1, y2]


X1, Y1 = np.meshgrid(x1, y1)
X1, Y1 = mkvc(X1), mkvc(Y1)
X2, Y2 = np.meshgrid(x2, y2)
X2, Y2 = mkvc(X2), mkvc(Y2)

# Plot data and tiles
rxLoc = survey.srcField.rxList[0].locs
# fig, ax1 = plt.figure(), plt.subplot()
# PF.Magnetics.plot_obs_2D(rxLoc,survey.dobs, ax=ax1)
# for ii in range(X1.shape[0]):
#     ax1.add_patch(Rectangle((X1[ii],Y1[ii]),X2[ii]-X1[ii],Y2[ii]-Y1[ii], facecolor = 'none', edgecolor='k'))
# ax1.set_aspect('equal')

# LOOP THROUGH TILES
npadxy = 8
core_x = np.ones(lx)*dx[0]
core_y = np.ones(lx)*dx[1]
expf = 1.3

os.system('if not exist ' + work_dir + out_dir + tile_dirl2 + ' mkdir ' + work_dir+out_dir+tile_dirl2)
os.system('if not exist ' + work_dir + out_dir + tile_dirlp + ' mkdir ' + work_dir+out_dir+tile_dirlp)

tiles = range(X1.shape[0])
tilestep = 2
for tt in tiles[tilestep:2*tilestep]:

    midX = np.mean([X1[tt], X2[tt]])
    midY = np.mean([Y1[tt], Y2[tt]])

    # Create new mesh
    padx = np.r_[dx[0]*expf**(np.asarray(range(npadxy))+1)]
    pady = np.r_[dx[1]*expf**(np.asarray(range(npadxy))+1)]

    hx = np.r_[padx[::-1], core_x, padx]
    hy = np.r_[pady[::-1], core_y, pady]
#        hz = np.r_[padb*2.,[33,26],np.ones(25)*22,[18,15,12,10,8,7,6], np.ones(18)*5,5*expf**(np.asarray(range(2*npad)))]
    hz = mesh.hz

    mesh_t = Mesh.TensorMesh([hx, hy, hz], 'CC0')

#    mtemp._x0 = [x0[ii]-np.sum(padb), y0[ii]-np.sum(padb), mesh.x0[2]]

    mesh_t._x0 = (mesh_t.x0[0] + midX, mesh_t.x0[1]+midY, mesh.x0[2])

    Mesh.TensorMesh.writeUBC(mesh_t, work_dir + out_dir + tile_dirl2 + "MVI_S_Tile" + str(tt) + ".msh")
    Mesh.TensorMesh.writeUBC(mesh_t, work_dir + out_dir + tile_dirlp + "MVI_S_Tile" + str(tt) + ".msh")

#        meshes.append(mtemp)
    # Grab the right data
    xlim = [mesh_t.vectorCCx[npadxy], mesh_t.vectorCCx[-npadxy]]
    ylim = [mesh_t.vectorCCy[npadxy], mesh_t.vectorCCy[-npadxy]]

    ind_t = np.all([rxLoc[:, 0] > xlim[0], rxLoc[:, 0] < xlim[1],
                    rxLoc[:, 1] > ylim[0], rxLoc[:, 1] < ylim[1]], axis=0)

    if np.sum(ind_t) < 20:
        continue

    rxLoc_t = PF.BaseMag.RxObs(rxLoc[ind_t, :])
    srcField = PF.BaseMag.SrcField([rxLoc_t], param=survey.srcField.param)
    survey_t = PF.BaseMag.LinearSurvey(srcField)
    survey_t.dobs = survey.dobs[ind_t]
    survey_t.std = survey.std[ind_t]

    # Extract model from global to local mesh
    if driver.topofile is not None:
        topo = np.genfromtxt(work_dir + driver.topofile,
                             skip_header=1)
        actv = Utils.surface2ind_topo(mesh_t, topo, 'N')
        actv = np.asarray(np.where(mkvc(actv))[0], dtype=int)
    else:
        actv = np.ones(mesh_t.nC, dtype='bool')

    nC = len(actv)
    print("Tile "+str(tt))
    print(nC, np.sum(ind_t))
    # Create active map to go from reduce space to full
    actvMap = Maps.InjectActiveCells(mesh_t, actv, 0)

    # Create identity map
    idenMap = Maps.IdentityMap(nP=3*nC)

    mstart = np.ones(3*len(actv))*1e-4

    # Create the forward model operator
    prob = PF.Magnetics.MagneticVector(mesh_t, chiMap=idenMap,
                                       actInd=actv)

    # Explicitely set starting model
    prob.chi = mstart

    # Pair the survey and problem
    survey_t.pair(prob)

    # Create sensitivity weights from our linear forward operator
    wr = np.sum(prob.F**2., axis=0)**0.5
    wr = (wr/np.max(wr))

    # Create a block diagonal regularization
    wires = Maps.Wires(('p', nC), ('s', nC), ('t', nC))

    # Create a regularization
    reg_p = Regularization.Sparse(mesh_t, indActive=actv, mapping=wires.p)
    reg_p.cell_weights = wires.p * wr
    reg_p.norms = [2, 2, 2, 2]

    reg_s = Regularization.Sparse(mesh_t, indActive=actv, mapping=wires.s)
    reg_s.cell_weights = wires.s * wr
    reg_s.norms = [2, 2, 2, 2]

    reg_t = Regularization.Sparse(mesh_t, indActive=actv, mapping=wires.t)
    reg_t.cell_weights = wires.t * wr
    reg_t.norms = [2, 2, 2, 2]

    reg = reg_p + reg_s + reg_t
    reg.mref = np.zeros(3*nC)

    # Data misfit function
    dmis = DataMisfit.l2_DataMisfit(survey_t)
    dmis.W = 1./survey_t.std

    # Add directives to the inversion
    opt = Optimization.ProjectedGNCG(maxIter=30, lower=-10., upper=10.,
                                     maxIterCG=20, tolCG=1e-3)

    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    betaest = Directives.BetaEstimate_ByEig()

    # Here is where the norms are applied
    IRLS = Directives.Update_IRLS(f_min_change=1e-4,
                                  minGNiter=3, beta_tol=1e-2)

    update_Jacobi = Directives.UpdatePreCond()
    targetMisfit = Directives.TargetMisfit()

    saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap)
    saveModel.fileName = work_dir + out_dir + 'MVI_C'
    inv = Inversion.BaseInversion(invProb,
                                  directiveList=[betaest, IRLS, update_Jacobi,
                                                 saveModel])

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
    reg_a = Regularization.Sparse(mesh_t, indActive=actv, mapping=wires.amp)
    reg_a.norms = driver.lpnorms[:4]
    if driver.eps is not None:
        reg_a.eps_p = driver.eps[0]
        reg_a.eps_q = driver.eps[1]
    else:
        reg_a.eps_p = np.percentile(np.abs(mstart[:nC]), 95)

    reg_t = Regularization.Sparse(mesh_t, indActive=actv, mapping=wires.theta)
    reg_t.alpha_s = 0.
    reg_t.space = 'spherical'
    reg_t.norms = driver.lpnorms[4:8]
    reg_t.eps_q = 2e-2
    # reg_t.alpha_x, reg_t.alpha_y, reg_t.alpha_z = 0.25, 0.25, 0.25

    reg_p = Regularization.Sparse(mesh_t, indActive=actv, mapping=wires.phi)
    reg_p.alpha_s = 0.
    reg_p.space = 'spherical'
    reg_p.norms = driver.lpnorms[8:]
    reg_p.eps_q = 1e-2

    reg = reg_a + reg_t + reg_p
    reg.mref = np.zeros(3*nC)

    # Data misfit function
    dmis = DataMisfit.l2_DataMisfit(survey_t)
    dmis.W = 1./survey_t.std

    # Add directives to the inversion
    opt = Optimization.ProjectedGNCG(maxIter=25,
                                     lower=[0., -np.inf, -np.inf],
                                     upper=[10., np.inf, np.inf],
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
    saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap)
    saveModel.fileName = work_dir + out_dir + 'MVI_S'

    inv = Inversion.BaseInversion(invProb,
                                  directiveList=[ProjSpherical, IRLS,
                                                 update_SensWeight,
                                                 update_Jacobi, saveModel])

    mrec_MVI_S = inv.run(mstart)

    mrec_MVI_S_xyz = PF.Magnetics.atp2xyz(mrec_MVI_S).reshape(nC,3,order='F')
    mx = actvMap*mrec_MVI_S_xyz[:,0]
    my = actvMap*mrec_MVI_S_xyz[:,1]
    mz = actvMap*mrec_MVI_S_xyz[:,2]
    PF.MagneticsDriver.writeVectorUBC(mesh_t,work_dir + out_dir + tile_dirl2 + "MVI_S_Tile"+ str(tt) +'.fld',np.c_[mx, my, mz])

    mrec_MVI_S_l2 = PF.Magnetics.atp2xyz(reg.l2model).reshape(nC,3,order='F')
    mx = actvMap*mrec_MVI_S_l2[:,0]
    my = actvMap*mrec_MVI_S_l2[:,1]
    mz = actvMap*mrec_MVI_S_l2[:,2]
    PF.MagneticsDriver.writeVectorUBC(mesh_t,work_dir + out_dir + tile_dirlp + "MVI_S_Tile"+ str(tt) +'.fld',np.c_[mx, my, mz])

    # Outputs
    PF.Magnetics.writeUBCobs(work_dir+out_dir + tile_dirlp + "MVI_S_Tile"+ str(tt) +"_Inv.pre", survey_t, invProb.dpred)


    del prob
    gc.collect()

#PF.Magnetics.plot_obs_2D(rxLoc,invProb.dpred,varstr='Amplitude Data')
