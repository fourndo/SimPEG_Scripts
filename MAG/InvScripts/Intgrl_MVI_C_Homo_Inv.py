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
work_dir = "C:\\Users\\DominiqueFournier\\Dropbox\\Projects\\Synthetic\\Block_Gaussian_topo\\"
#work_dir = "C:\\Egnyte\\Private\\dominiquef\\Projects\\4559_CuMtn_ZTEM\\Modeling\\MAG\\A1_Fenton\\"
#work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Synthetic\\Nut_Cracker\\"
# work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\TKC\\DIGHEM_TMI\\"
# work_dir = "C:\\Users\\DominiqueFournier\\Documents\\GIT\\InnovationGeothermal\\FORGE\\SyntheticModel\\"
#work_dir = "C:\\Users\\DominiqueFournier\\Desktop\\Demo\\"
# work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Yukon\\Modeling\\MAG\\"

out_dir = "SimPEG_MVI_C_HomogenInv\\"
input_file = "SimPEG_MAG.inp"
# dwnSamplingFact = 0.1
# %%
# Read in the input file which included all parameters at once
# (mesh, topo, model, survey, inv param, etc.)
driver = PF.MagneticsDriver.MagneticsDriver_Inv(work_dir + input_file)

os.system('if not exist ' + work_dir + out_dir + ' mkdir ' + work_dir+out_dir)

# Access the mesh and survey information
mesh = driver.mesh
survey = driver.survey
actv = driver.activeCells

#nD = int(survey.nD*dwnSamplingFact)
#print("nD ratio:" + str(nD) + '\\' + str(survey.nD))
#indx = np.random.randint(0, high=survey.nD, size=nD)
## Create a new downsampled survey
#locXYZ = survey.srcField.rxList[0].locs[indx, :]
#
#dobs = survey.dobs
#std = survey.std
#
#rxLoc = PF.BaseGrav.RxObs(locXYZ)
#srcField = PF.BaseMag.SrcField([rxLoc], param=survey.srcField.param)
#survey = PF.BaseMag.LinearSurvey(srcField)
#survey.dobs = dobs[indx]
#survey.std = std[indx]


# WORK IN PROGRESS
# NEED TO CREATE A MAPPING FROM UNITSs
# Load the starting model (with susceptibility)
# Load the reference model (with geological definition)
msus = driver.m0  # Starting model
mgeo = driver.mref

# Get unique geo units
geoUnits = np.unique(mgeo).tolist()


# Build a dictionary for the units
#medVal = np.asarray([np.median(msus[mgeo==unit]) for unit in geoUnits])
#actv = msus!=-100

# Build list of indecies for the geounits
index = []
for unit in geoUnits:
    if unit != 0:
        index += [mgeo[actv] == unit]
nC = len(index)
# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, 0)

# Creat reduced identity map
homogMap = Maps.SurjectUnits(index)
homogMap.nBlock = 3

# Create the forward model operator
prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=homogMap, actInd=actv, modelType='vector')

# Pair the survey and problem
survey.pair(prob)

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std

# Create sensitivity weights from our linear forward operator
rxLoc = survey.srcField.rxList[0].locs
#wr = np.zeros(3*nC)
m0 = np.ones(3*nC)*1e-6
#for ii in range(survey.nD):
#    wr += ((prob.F[ii, :]*prob.mapping.deriv(m0))/survey.std[ii])**2.
wr = prob.getJtJdiag(m0, W=dmis.W)
wr = (wr/np.max(wr))
wr = wr**0.5

#Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + 'SensWeights.sus',
#                              actvMap*(homogMap.P*wr))

# Create a block diagonal regularization
wires = Maps.Wires(('p', nC), ('s', nC), ('t', nC))

idenMap = Maps.IdentityMap(nP=nC)

regMesh = Mesh.TensorMesh([nC])

# Create a regularization
reg_p = Regularization.Sparse(regMesh, mapping=wires.p)
#reg_p.cell_weights = wires.p * wr
#reg_p.norms = [2, 2, 2, 2]

reg_s = Regularization.Sparse(regMesh, mapping=wires.s)
#reg_s.cell_weights = wires.s * wr
#reg_s.norms = [2, 2, 2, 2]

reg_t = Regularization.Sparse(regMesh, mapping=wires.t)
#reg_t.cell_weights = wires.t * wr
#reg_t.norms = [2, 2, 2, 2]

reg = reg_p + reg_s + reg_t
reg.mref = np.zeros(3*nC)



# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=10, lower=-10., upper=10.,
                                 maxIterCG=20, tolCG=1e-3)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
betaest = Directives.BetaEstimate_ByEig()

# Here is where the norms are applied
IRLS = Directives.Update_IRLS(f_min_change=1e-4,
                              minGNiter=3, beta_tol=1e-2)

update_Jacobi = Directives.UpdatePreconditioner()
targetMisfit = Directives.TargetMisfit()

saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap)
saveModel.fileName = work_dir + out_dir + 'MVI_C'
inv = Inversion.BaseInversion(invProb,
                              directiveList=[betaest, IRLS, update_Jacobi])

mrec_MVI = inv.run(m0)

mrec_MVI = (homogMap.P*mrec_MVI).reshape((np.sum(actv), 3), order='F')
mx = actvMap*mrec_MVI[:, 0]
my = actvMap*mrec_MVI[:, 1]
mz = actvMap*mrec_MVI[:, 2]
Utils.io_utils.writeVectorUBC(mesh, work_dir + out_dir + "MVIC" + '.fld',
                                  np.c_[mx, my, mz])

# Outputs
Utils.io_utils.writeUBCmagneticsObservations(work_dir+out_dir + "MVIC" + "_Inv.pre", survey,
                         invProb.dpred)
