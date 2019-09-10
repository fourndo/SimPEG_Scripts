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
from pymatsolver import PardisoSolver

#work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Kevitsa\\Modeling\\MAG\\"
#work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\\Research\\Synthetic\\Block_Gaussian_topo\\"
# work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Synthetic\\SingleBlock\\Simpeg\\"
#work_dir = "C:\\Egnyte\\Private\\dominiquef\\Projects\\4559_CuMtn_ZTEM\\Modeling\\MAG\\A1_Fenton\\"
work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Synthetic\\Nut_Cracker\\"
# work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\TKC\\DIGHEM_TMI\\"
#work_dir = "C:\\Users\\DominiqueFournier\\Documents\\GIT\\InnovationGeothermal\\FORGE\\"
#work_dir = "C:\\Users\\DominiqueFournier\\Desktop\\Demo\\"

out_dir = "SimPEG_Susc_HomogenInv\\"
input_file = "SimPEG_MAG.inp"
# %%
# Read in the input file which included all parameters at once
# (mesh, topo, model, survey, inv param, etc.)
driver = PF.MagneticsDriver.MagneticsDriver_Inv(work_dir + input_file)

os.system('if not exist ' + work_dir + out_dir + ' mkdir ' + work_dir+out_dir)

# Access the mesh and survey information
mesh = driver.mesh
survey = driver.survey
actv = driver.activeCells

# WORK IN PROGRESS
# NEED TO CREATE A MAPPING FROM UNITSs
# Load the starting model (with susceptibility)
# Load the reference model (with geological definition)
msus = driver.m0  # Starting model
mgeo = driver.mref

# Get unique geo units
geoUnits = np.unique(mgeo).tolist()


# Build a dictionary for the units
# medVal = np.asarray([np.median(msus[mgeo==unit]) for unit in geoUnits])


actv = msus!=-100
# Build list of indecies for the geounits
index = []
for unit in geoUnits:
    if unit!=0:
        index += [mgeo[actv]==unit]
nC = len(index)
m0 = np.ones(nC)*1e-6
# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

# Creat reduced identity map
homogMap = Maps.Homogenize(index)

# Create the forward model operator
prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=homogMap, actInd=actv, Solver=PardisoSolver)

# Pair the survey and problem
survey.pair(prob)

# Create sensitivity weights from our linear forward operator
rxLoc = survey.srcField.rxList[0].locs
wr = np.zeros(nC)
for ii in range(survey.nD):
    wr += ((prob.F[ii, :]*prob.chiMap.deriv(m0))/survey.std[ii])**2.

wr = (wr/np.max(wr))
wr = wr**0.5

Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + 'SensWeights.sus',
                              actvMap*(homogMap.P*wr))
# wr = PF.Magnetics.get_dist_wgt(mesh, rxLoc, actv, 3, 1)

# Create reduced identity map
idenMap = Maps.IdentityMap(nP=nC)

regMesh = Mesh.TensorMesh([nC])
# Create a regularization
reg = Regularization.Sparse(regMesh, mapping=idenMap)
reg.norms = driver.lpnorms

if driver.eps is not None:
    reg.eps_p = driver.eps[0]
    reg.eps_q = driver.eps[1]

reg.cell_weights = wr#driver.cell_weights*mesh.vol**0.5
reg.mref = np.zeros(nC)
# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=25, lower=0., upper=10.,
                                 maxIterLS=20, maxIterCG=10, tolCG=1e-4)
invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
betaest = Directives.BetaEstimate_ByEig()

# Here is where the norms are applied
# Use pick a treshold parameter empirically based on the distribution of
#  model parameters
IRLS = Directives.Update_IRLS(f_min_change=1e-3, minGNiter=3, maxIRLSiter=10)
update_Jacobi = Directives.UpdateJacobiPrecond()

# saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap)
# saveModel.fileName = work_dir + out_dir + 'ModelSus'

inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, IRLS, update_Jacobi])


# prob.model = m0
mrec = inv.run(m0)

Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + 'ModelSus_lp.sus',
                              actvMap*(homogMap.P*mrec))

PF.Magnetics.writeUBCobs(work_dir + out_dir + 'InvPredicted.pre', survey, invProb.dpred)
if getattr(invProb, 'l2model', None) is not None:
    Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + 'ModelSus_l2.sus',
                                  actvMap*(homogMap.P*invProb.l2model))


