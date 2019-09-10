from SimPEG import Mesh, Utils, Maps, PF, Regularization, Optimization, Directives, DataMisfit, InvProblem, Inversion
from SimPEG.Utils import mkvc, sdiag
import numpy as np
import scipy as sp
import os


#work_dir = 'C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Modelling\\Synthetic\\SingleBlock\\GRAV'
#work_dir = 'C:\\Users\\DominiqueFournier\\Downloads'
work_dir = 'C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Kevitsa\\Modeling\\GRAV'
inpfile = 'SimPEG_GRAV.inp'
out_dir = "SimPEG_PF_Inv\\"
dsep = '\\'
dsep = os.path.sep

os.system('mkdir ' + work_dir + dsep + out_dir)


# Read input file
driver = PF.GravityDriver.GravityDriver_Inv(work_dir + dsep + inpfile)
mesh = driver.mesh
survey = driver.survey

# Create a sub-survey and sub-sample randomly
nD = int(survey.dobs.shape[0]*0.05)
print("nD ratio:" + str(nD) +'\\' + str(survey.dobs.shape[0]) )
indx = np.random.randint(0, high=survey.dobs.shape[0], size=nD)
# Create a new downsampled survey
locXYZ = survey.srcField.rxList[0].locs[indx,:]

dobs = survey.dobs
std = survey.std

rxLoc = PF.BaseGrav.RxObs(locXYZ)
srcField = PF.BaseMag.SrcField([rxLoc], param=survey.srcField.param)
survey = PF.BaseMag.LinearSurvey(srcField)
survey.dobs = dobs[indx]
survey.std = std[indx]

#rxLoc = survey.srcField.rxList[0].locs
d = survey.dobs
wd = survey.std

# SUB FUNCTIONS
from SimPEG.Utils import sdiag, speye, kron3
import matplotlib.gridspec as gridspec

def Rz(theta):
    """Rotation matrix about z axis"""
    Rz = np.vstack((np.r_[np.cos((theta)),
                          -np.sin((theta)), 0],
                   np.r_[np.sin((theta)),
                         np.cos((theta)), 0],
                   np.r_[0, 0, 1]))
    return Rz


def Ry(theta):
    """Rotation matrix about y axis"""
    Ry = np.vstack( (np.r_[np.cos((theta)), 0,
                     np.sin((theta))],
                   np.r_[0, 1, 0],
               np.r_[-np.sin((theta)), 0,
                     np.cos((theta))]))

    return Ry


def ddx(n,vals):
    """Define 1D averaging operator from cell-centers to nodes."""
    ddx = (
        sp.sparse.spdiags(
            (np.ones((n, 1)) * vals).T,
            [-1, 0, 1],
            n , n,
            format="csr"
        )
    )
    ddx[-1,:] = ddx[-2,:]
    return ddx


def getDiffOpRot(mesh, psi, theta, phi, vec, forward = True):

    hz = np.kron(mesh.hz, np.kron(np.ones(mesh.vnC[1]), np.ones(mesh.vnC[0])))
    hy = np.kron(np.ones(mesh.vnC[2]), np.kron(mesh.hy, np.ones(mesh.vnC[0])))
    hx = np.kron(np.ones(mesh.vnC[2]), np.kron(np.ones(mesh.vnC[1]), mesh.hx))

    unitMesh = Mesh.TensorMesh([np.ones(3),np.ones(3),np.ones(3)], x0='CCC')

    stencil = []
    for ii in range(unitMesh.nC):
        stencil += [np.kron(np.r_[-1,1],[0.5, 0.5, 0.5]).reshape((2,3)) +
                    np.kron(np.ones(2),unitMesh.gridCC[ii,:]).reshape((2,3))]


    if isinstance(theta, float):
        theta = np.ones(mesh.nC) * theta

    if isinstance(phi, float):
        phi = np.ones(mesh.nC) * phi


    if isinstance(psi, float):
        psi = np.ones(mesh.nC) * psi

    if forward:
        ind = 1
    else:
        ind = -1

    if vec=='X':
        px = np.kron(np.ones(mesh.nC),np.c_[ind,0,0])
        theta = np.arctan2((np.sin(theta)/hz),(np.cos(theta)/hx))
        phi = np.arctan2((np.sin(phi)/hy),(np.cos(phi)/hx))
        psi = np.arctan2((np.sin(psi)/hz),(np.cos(psi)/hy))

    elif vec=='Y':
        px = np.kron(np.ones(mesh.nC),np.c_[0,ind,0])
        theta = np.arctan2((np.sin(theta)/hz),(np.cos(theta)/hx))
        phi = np.arctan2((np.sin(phi)/hx),(np.cos(phi)/hy))
        psi = np.arctan2((np.sin(psi)/hz),(np.cos(psi)/hy))

    else:
        px = np.kron(np.ones(mesh.nC),np.c_[0,0,ind])
        theta = np.arctan2((np.sin(theta)/hx),(np.cos(theta)/hz))
        phi = np.arctan2((np.sin(phi)/hy),(np.cos(phi)/hx))
        psi = np.arctan2((np.sin(psi)/hy),(np.cos(psi)/hz))

    #     v = np.ones((mesh.nC,27))
    #     for ii in range(mesh.nC):

#     S = Utils.sdiag(mkvc(np.c_[1/ratiox**0.5, 1/ratioy**0.5, 1/ratioz**0.5]))

    # Create sparse rotation operators
    rxa = mkvc(np.c_[np.ones(mesh.nC), np.cos(psi), np.cos(psi)].T)
    rxb = mkvc(np.c_[np.zeros(mesh.nC), np.sin(psi),np.zeros(mesh.nC)].T)
    rxc = mkvc(np.c_[np.zeros(mesh.nC), -np.sin(psi),np.zeros(mesh.nC)].T)
    Rx = sp.sparse.diags([rxb[:-1],rxa,rxc[:-1]],[-1,0,1])

    rya = mkvc(np.c_[np.cos(theta), np.ones(mesh.nC), np.cos(theta)].T)
    ryb = mkvc(np.c_[-np.sin(theta), np.zeros(mesh.nC), np.zeros(mesh.nC)].T)
    ryc = mkvc(np.c_[np.sin(theta), np.zeros(mesh.nC), np.zeros(mesh.nC)].T)
    Ry = sp.sparse.diags([ryb[:-2],rya,ryc[:-2]],[-2,0,2])

    rza = mkvc(np.c_[np.cos(phi), np.cos(phi),np.ones(mesh.nC)].T)
    rzb = mkvc(np.c_[np.sin(phi), np.zeros(mesh.nC), np.zeros(mesh.nC)].T)
    rzc = mkvc(np.c_[-np.sin(phi), np.zeros(mesh.nC), np.zeros(mesh.nC)].T)
    Rz = sp.sparse.diags([rzb[:-1],rza,rzc[:-1]],[-1,0,1])

    # Rotate all cell vectors
    rx = (Rz*(Ry*(Rx*px.T))).reshape((mesh.nC,3))

    # Move the bottom-SW and top-NE nodes
    nBSW = np.kron(stencil[13][0],np.ones((mesh.nC,1)))+rx
    nTNE = np.kron(stencil[13][1],np.ones((mesh.nC,1)))+rx

    # Compute fractional volumes with base stencil
    V=[]
    for s in stencil:

        sBSW = np.kron(s[0],np.ones((mesh.nC,1)))
        sTNE = np.kron(s[1],np.ones((mesh.nC,1)))

        V += [(np.max([np.min([sTNE[:, 0], nTNE[:, 0]], axis=0)-
                   np.max([sBSW[:, 0], nBSW[:, 0]], axis=0), np.zeros(mesh.nC)], axis=0) *
           np.max([np.min([sTNE[:, 1], nTNE[:, 1]], axis=0)-
                   np.max([sBSW[:, 1], nBSW[:, 1]], axis=0), np.zeros(mesh.nC)], axis=0) *
           np.max([np.min([sTNE[:, 2], nTNE[:, 2]], axis=0)-
                   np.max([sBSW[:, 2], nBSW[:, 2]], axis=0), np.zeros(mesh.nC)], axis=0))]

    count = -1
    Gx = speye(mesh.nC)

    for ii in range(3):
        flagz = [0,0,0]
        flagz[ii] = 1

        for jj in range(3):
            flagy = [0,0,0]
            flagy[jj] = 1

            for kk in range(3):

                flagx = [0,0,0]
                flagx[kk] = 1

                count += 1
                Gx -= sdiag(np.ones(mesh.nC)*V[count])*kron3( ddx(mesh.nCz,flagz), ddx(mesh.nCy,flagy), ddx(mesh.nCx,flagx) )

    return Gx, rx

def plotModelSections(mesh, m, normal='x', ind=0, vmin=None, vmax=None,
                      subFact=2, scale=1., xlim=None, ylim=None, vec='k',
                      title=None, axs=None, ndv=-100, contours=None, fill=True,
                      orientation='vertical', cmap='pink_r'):

    """
    Plot section through a 3D tensor model
    """
    # plot recovered model
    nC = mesh.nC

    if vmin is None:
        vmin = m.min()

    if vmax is None:
        vmax = m.max()

    if len(m) == 3*nC:
        m_lpx = m[0:nC]
        m_lpy = m[nC:2*nC]
        m_lpz = m[2*nC:]

        m_lpx[m_lpx == ndv] = np.nan
        m_lpy[m_lpy == ndv] = np.nan
        m_lpz[m_lpz == ndv] = np.nan

        amp = np.sqrt(m_lpx**2. + m_lpy**2. + m_lpz**2.)

        m_lpx = (m_lpx).reshape(mesh.vnC, order='F')
        m_lpy = (m_lpy).reshape(mesh.vnC, order='F')
        m_lpz = (m_lpz).reshape(mesh.vnC, order='F')
        amp = amp.reshape(mesh.vnC, order='F')
    else:
        m[m == ndv] = np.nan
        amp = m.reshape(mesh.vnC, order='F')

    xx = mesh.gridCC[:, 0].reshape(mesh.vnC, order="F")
    zz = mesh.gridCC[:, 2].reshape(mesh.vnC, order="F")
    yy = mesh.gridCC[:, 1].reshape(mesh.vnC, order="F")

    if axs is None:
        fig, axs = plt.figure(), plt.subplot()

    if normal == 'x':
        xx = yy[ind, :, :].T
        yy = zz[ind, :, :].T
        model = amp[ind, :, :].T

        if len(m) == 3*nC:
            mx = m_lpy[ind, ::subFact, ::subFact].T
            my = m_lpz[ind, ::subFact, ::subFact].T

    elif normal == 'y':
        xx = xx[:, ind, :].T
        yy = zz[:, ind, :].T
        model = amp[:, ind, :].T

        if len(m) == 3*nC:
            mx = m_lpx[::subFact, ind, ::subFact].T
            my = m_lpz[::subFact, ind, ::subFact].T

    elif normal == 'z':

        actIndFull = np.zeros(mesh.nC, dtype=bool)
        actIndFull[actv] = True

        actIndFull = actIndFull.reshape(mesh.vnC, order='F')

        model = np.zeros((mesh.nCx,mesh.nCy))
        mx = np.zeros((mesh.nCx,mesh.nCy))
        my = np.zeros((mesh.nCx,mesh.nCy))
        for ii in range(mesh.nCx):
            for jj in range(mesh.nCy):

                zcol = actIndFull[ii, jj, :]
                model[ii, jj] = amp[ii, jj, np.where(zcol)[0][-ind]]

                if len(m) == 3*nC:
                    mx[ii, jj] = m_lpx[ii, jj, np.where(zcol)[0][-ind]]
                    my[ii, jj] = m_lpy[ii, jj, np.where(zcol)[0][-ind]]

        xx = xx[:, :, ind].T
        yy = yy[:, :, ind].T
        model = model.T

        if len(m) == 3*nC:
            mx = mx[::subFact, ::subFact].T
            my = my[::subFact, ::subFact].T

    im2, cbar =[], []
    if fill:
        im2 = axs.contourf(xx, yy, model,
                           30, vmin=vmin, vmax=vmax, clim=[vmin, vmax],
                           cmap=cmap)

        cbar = plt.colorbar(im2, orientation=orientation, ax=axs,
                 ticks=np.linspace(im2.vmin, im2.vmax, 4),
                 format="${%.3f}$", shrink=0.5)
    if contours is not None:
        axs.contour(xx, yy, model, contours, colors='k')

    if len(m) == 3*nC:

        axs.quiver(mkvc(xx[::subFact, ::subFact]),
                   mkvc(yy[::subFact, ::subFact]),
                   mkvc(mx),
                   mkvc(my),
                   pivot='mid',
                   scale_units="inches", scale=scale, linewidths=(1,),
                   edgecolors=(vec),
                   headaxislength=0.1, headwidth=10, headlength=30)

    axs.set_aspect('equal')

    if xlim is not None:
        axs.set_xlim(xlim[0], xlim[1])

    if ylim is not None:
        axs.set_ylim(ylim[0], ylim[1])

    if title is not None:
        axs.set_title(title)

    return axs, im2, cbar

# Generate active cell and starting model
ndata = survey.srcField.rxList[0].locs.shape[0]

actv = driver.activeCells
nC = len(actv)

# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

# Create static map
static = driver.staticCells
dynamic = driver.dynamicCells

staticCells = Maps.InjectActiveCells(None, dynamic, driver.m0[static], nC=nC)
mstart = driver.m0[dynamic]

prob = PF.Gravity.GravityIntegral(mesh, rhoMap=staticCells, actInd=actv)
prob.solverOpts['accuracyTol'] = 1e-4

survey.pair(prob)

nC = prob.G.shape[1]

# Write out the predicted file and generate the forward operator
pred = prob.fields(mstart)

PF.Gravity.writeUBCobs(work_dir + dsep + 'Pred0.dat',survey,pred)

# Load in control points for the rotation
mk = np.loadtxt(work_dir + dsep + 'Structural_Constraints_Surfs.dat')
pts, vals = mk[:, :3], mk[:, 3:]

# Interpolate dip/azm in 3D
mInterp = Utils.modelutils.MinCurvatureInterp(mesh, pts, vals, tol=1e-5,
                                              iterMax=500)
mout = sdiag(np.sum(mInterp[:, :3]**2, axis=1)**-0.5) * mInterp[:, :3]

# # Interpolate dip/azm in 3D
mLp = mInterp[:, 3:]


# Convert normals to rotation angles
atp = Utils.coordutils.xyz2atp(mout)

theta = atp[mesh.nC:2*mesh.nC]
phi = atp[2*mesh.nC:]
psi = np.ones_like(theta)*np.deg2rad(0.)

indActive = np.zeros(mesh.nC, dtype=bool)
indActive[actv] = True


# This is for the aircells
Pac = Utils.speye(mesh.nC)[:, indActive]

Dx1, rx1 = getDiffOpRot(mesh, psi, theta, phi, 'X')
Dy1, ry1 = getDiffOpRot(mesh, psi, theta, phi, 'Y')
Dz1, rz1 = getDiffOpRot(mesh, psi, theta, phi, 'Z')

Dx2, rx2 = getDiffOpRot(mesh, psi, theta, phi, 'X', forward=False)
Dy2, ry2 = getDiffOpRot(mesh, psi, theta, phi, 'Y', forward=False)
Dz2, rz2 = getDiffOpRot(mesh, psi, theta, phi, 'Z', forward=False)

Dx1 = Pac.T * Dx1 * Pac
Dy1 = Pac.T * Dy1 * Pac
Dz1 = Pac.T * Dz1 * Pac

Dx2 = Pac.T * Dx2 * Pac
Dy2 = Pac.T * Dy2 * Pac
Dz2 = Pac.T * Dz2 * Pac


# Load weighting  file
if driver.wgtfile is None:
    # wr = PF.Magnetics.get_dist_wgt(mesh, rxLoc, actv, 3., np.min(mesh.hx)/4.)
    # wr = wr**2.

    # Make depth weighting
    wr = np.sum(prob.G**2., axis=0)**0.5
    wr = (wr/np.max(wr))
    # wr_out = actvMap * wr

else:
    wr = Mesh.TensorMesh.readModelUBC(mesh, work_dir + dsep + wgtfile)
    wr = wr[actv]
    wr = wr**2.

# % Create inversion objects
reg1 = Regularization.Sparse(mesh, indActive=actv, mapping=staticCells)
reg1.mref = driver.mref[dynamic]
reg1.cell_weights = wr
reg1.norms = driver.lpnorms
if driver.eps is not None:
    reg1.eps_p = driver.eps[0]
    reg1.eps_q = driver.eps[1]

reg1.objfcts[1].regmesh._cellDiffxStencil = Dx1
reg1.objfcts[1].regmesh._aveCC2Fx = speye(nC)

reg1.objfcts[2].regmesh._cellDiffyStencil = Dy1
reg1.objfcts[2].regmesh._aveCC2Fy = speye(nC)

reg1.objfcts[3].regmesh._cellDiffzStencil = Dz1
reg1.objfcts[3].regmesh._aveCC2Fz = speye(nC)

reg2 = Regularization.Sparse(mesh, indActive=actv, mapping=staticCells)
reg2.mref = driver.mref[dynamic]
reg2.cell_weights = wr
reg2.norms = driver.lpnorms
if driver.eps is not None:
    reg2.eps_p = driver.eps[0]
    reg2.eps_q = driver.eps[1]

reg2.objfcts[1].regmesh._cellDiffxStencil = Dx2
reg2.objfcts[1].regmesh._aveCC2Fx = speye(nC)

reg2.objfcts[2].regmesh._cellDiffyStencil = Dy2
reg2.objfcts[2].regmesh._aveCC2Fy = speye(nC)

reg2.objfcts[3].regmesh._cellDiffzStencil = Dz2
reg2.objfcts[3].regmesh._aveCC2Fz = speye(nC)

reg= reg1 + reg2

#%%
opt = Optimization.ProjectedGNCG(maxIter=100, lower=driver.bounds[0],
                                 upper=driver.bounds[1], maxIterLS = 20,
                                 maxIterCG= 10, tolCG = 1e-3)
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./wd
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

betaest = Directives.BetaEstimate_ByEig()
IRLS = Directives.Update_IRLS(f_min_change=1e-4, minGNiter=3)
update_Jacobi = Directives.Update_lin_PreCond()
#saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap)
#saveModel.fileName = work_dir + dsep + out_dir + 'GRAV'
inv = Inversion.BaseInversion(invProb, directiveList=[betaest, IRLS,
                                                      update_Jacobi])
# Run inversion
m0 = np.ones(actv.shape[0])*1e-4  # Starting model
prob.model = m0
mrec = inv.run(m0)

# Plot predicted
pred = prob.fields(mrec)

survey.dobs = pred
# PF.Gravity.plot_obs_2D(survey, 'Observed Data')
print("Final misfit:" + str(np.sum(((d-pred)/wd)**2.)))

m_out = actvMap*staticCells*IRLS.l2model

# Write result
Mesh.TensorMesh.writeModelUBC(mesh, work_dir + dsep + out_dir + 'SimPEG_inv_l2l2.den',m_out)

m_out = actvMap*staticCells*mrec
# Write result
Mesh.TensorMesh.writeModelUBC(mesh, work_dir + dsep + out_dir + 'SimPEG_inv_lplq.den',m_out)

