import matplotlib.pyplot as plt
import numpy as np

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

from discretize.utils import closestPoints
from SimPEG.Utils import mkvc
from scipy.spatial import cKDTree
import multiprocessing
import time
import dask

@dask.delayed
def makeSubProblem(args):
    globalMesh, globalActive, globalSurvey, globalTree, ind, h, padDist = args

    rxLoc = globalSurvey.srcField.rxList[0].locs

    loc = np.c_[rxLoc[ind, :]].T
    rx = PF.BaseMag.RxObs(loc)

    srcField = PF.BaseMag.SrcField([rx], param=globalSurvey.srcField.param)
    survey_t = PF.BaseMag.LinearSurvey(srcField)
    survey_t.dobs = np.c_[globalSurvey.dobs[ind]]
    survey_t.std = np.c_[globalSurvey.std[ind]]
    survey_t.index = ind

    # Create a mesh

    # Keep same fine cells as global
    h = [globalMesh.hx.min(), globalMesh.hy.min(), globalMesh.hz.min()]


    mesh_t = Utils.modelutils.meshBuilder(
        rxLoc, h, padDist, meshType='TREE', meshGlobal=globalMesh,
        verticalAlignment='center'
    )

    # Refine the mesh around loc
    mesh_t = Utils.modelutils.refineTree(
        mesh_t, loc, dtype='point',
        nCpad=[3, 3, 3], finalize=True
    )
    actv_t = np.ones(mesh_t.nC, dtype='bool')

    # Create reduced identity map
    tileMap = Maps.Tile((globalMesh, globalActive), (mesh_t, actv_t))
    tileMap._tree = globalTree

    # Create the forward model operator
    prob_t = PF.Magnetics.MagneticIntegral(
            mesh_t, chiMap=tileMap, actInd=actv_t,
            verbose=False)

    survey_t.pair(prob_t)

    # Pre-calc sensitivities and projections
    prob_t.G
    tileMap.P

    # Data misfit function
    dmis = DataMisfit.l2_DataMisfit(survey_t, eps=survey_t.std)
    dmis.W = 1./survey_t.std

    return dmis


if __name__ == '__main__':

    H0 = (50000, 90, 0)

    # Create a mesh
    dx = 5.

    hxind = [(dx, 5, -1.3), (dx, 10), (dx, 5, 1.3)]
    hyind = [(dx, 5, -1.3), (dx, 20), (dx, 5, 1.3)]
    hzind = [(dx, 5, -1.3), (dx, 10)]

    mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CC0')
    mesh.x0 = mesh.x0 - np.r_[0, 0, mesh.gridN[-1,2]]

    # Get index of the center
    midx = int(mesh.nCx/2)
    midy = int(mesh.nCy/2)

    # Lets create a simple Gaussian topo and set the active cells
    [xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
    zz = -np.exp((xx**2 + yy**2) / 75**2) + mesh.vectorNz[-1]

    # We would usually load a topofile
    topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]

    # Go from topo to actv cells
    actv = Utils.surface2ind_topo(mesh, topo, 'N')
    actv = np.asarray([inds for inds, elem in enumerate(actv, 1) if elem],
                      dtype=int) - 1
    # actv = np.ones(mesh.nC, dtype='bool')

    # Create active map to go from reduce space to full
    actvMap = Maps.InjectActiveCells(mesh, actv, -100)
    nC = len(actv)

    # Create and array of observation points
    xr = np.linspace(-22.5, 22.5, 10)
    yr = np.linspace(-32.5, 32.5, 10)
    X, Y = np.meshgrid(xr, yr)

    # Move the observation points 5m above the topo
    Z = -np.exp((X**2 + Y**2) / 75**2) + mesh.vectorNz[-1] + 5. # np.ones_like(X)*mesh.vectorNz[-1] + 5.  #

    # Create a MAGsurvey
    rxLoc = np.c_[Utils.mkvc(X.T), Utils.mkvc(Y.T), Utils.mkvc(Z.T)]
    rx = PF.BaseMag.RxObs(rxLoc)
    srcField = PF.BaseMag.SrcField([rx], param=H0)
    survey = PF.BaseMag.LinearSurvey(srcField)

    # We can now create a susceptibility model and generate data
    model = np.zeros(mesh.nC)

    # Add a block in half-space
    model = Utils.ModelBuilder.addBlock(
        mesh.gridCC, model, np.r_[-10, -30, -30], np.r_[10, -10, -10], 0.05
    )
    model = Utils.ModelBuilder.addBlock(
        mesh.gridCC, model, np.r_[-10, 10, -30], np.r_[10, 30, -10], 0.05
    )
    model = Utils.ModelBuilder.addBlock(
        mesh.gridCC, model, np.r_[50, 50, -30], np.r_[120, 120, -10], 0.2
    )

    model = Utils.mkvc(model)
    model = model[actv]

    # Create active map to go from reduce set to full
    actvMap = Maps.InjectActiveCells(mesh, actv, -100)

    # Creat reduced identity map
    idenMap = Maps.IdentityMap(nP=nC)

    # Create the forward model operator
    prob = PF.Magnetics.MagneticIntegral(
            mesh, chiMap=idenMap, actInd=actv,
            parallelized=False)

    # Pair the survey and problem
    survey.pair(prob)

    # Compute linear forward operator and compute some data
    d = prob.fields(model)

    # Add noise and uncertainties
    # We add some random Gaussian noise (1nT)
    data = d + np.random.randn(len(d))
    wd = np.ones(len(data))*1.  # Assign flat uncertainties

    survey.dobs = data
    survey.std = wd
    survey.mtrue = model

    # Plot the data
    rxLoc = survey.srcField.rxList[0].locs

    tree = cKDTree(mesh.gridCC[actv, :])

    h = [5., 5., 5.]
    padDist = np.ones((3, 2)) * 100
    wr = np.zeros(actv.sum())
    dmis = []

    print("Building Sub-problems")
#    pool = multiprocessing.Pool(4)
#    print(rxLoc.shape)
#    objfcts = pool.map(
#            makeSubProblem,
#            [
#             (mesh, actv, survey, tree, ii, h, padDist)
#             for ii in range(rxLoc.shape[0])
#            ]
#    )
#    pool.close()
#    pool.join()


    objfcts = {}
    for ii in range(rxLoc.shape[0]):

            objfcts[ii] = makeSubProblem((mesh, actv, survey, tree, ii, h, padDist))

    dask.compute(objfcts, num_workers=3)

    globalMisfit = objfcts[0]
    wr = mkvc((objfcts[0].prob.G * objfcts[0].prob.chiMap.deriv(None))**2.)


    print("Computing sensitivity cell_weights")
    for objfct in objfcts[1:]:

        globalMisfit += objfct
        wr += mkvc((objfct.prob.G*objfct.prob.chiMap.deriv(None))**2.)

    wr /= wr.max()
    wr **= 0.5

    # Create a regularization
    reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
    reg.cell_weights = wr
    # reg.norms = [0, 1, 1, 1]
    # reg.eps_p, reg.eps_q = 1e-3, 1e-3

    mesh.writeModelUBC('J1.dat', actvMap*wr)
    mesh.writeModelUBC('J.dat', actvMap*np.sum(prob.G**2., axis=0))

    # Add directives to the inversion
    opt = Optimization.ProjectedGNCG(maxIter=100, lower=0., upper=1.,
                                     maxIterLS=20, maxIterCG=10, tolCG=1e-3)
    invProb = InvProblem.BaseInvProblem(globalMisfit, reg, opt)
    betaest = Directives.BetaEstimate_ByEig()

    # Here is where the norms are applied
    # Use pick a treshold parameter empirically based on the distribution of
    #  model parameters
    IRLS = Directives.Update_IRLS(f_min_change=1e-3, minGNiter=3)
    update_Jacobi = Directives.UpdatePreconditioner()
    inv = Inversion.BaseInversion(invProb,
                                  directiveList=[IRLS, betaest, update_Jacobi])

    # Run the inversion
    m0 = np.ones(nC)*1e-4  # Starting model
    mrec = inv.run(m0)

    mesh.writeModelUBC('Mrec.sus', actvMap*mrec)
