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

out_dir = "SimPEG_MVI_S_HomogenInv\\"
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
medVal = np.asarray([np.median(msus[mgeo==unit]) for unit in geoUnits])


actv = msus!=-100
# Build list of indecies for the geounits
index = []
for unit in geoUnits:
    if unit!=0:
        index += [mgeo[actv]==unit]
nC = len(index)
# Create active map to go from reduce set to full
actvMap = Maps.InjectActiveCells(mesh, actv, -100)

# Creat reduced identity map
homogMap = Maps.Homogenize(index)
homogMap.nBlock = 3

# Create the forward model operator
prob = PF.Magnetics.MagneticVector(mesh, chiMap=homogMap, actInd=actv)

# Pair the survey and problem
survey.pair(prob)

# Create sensitivity weights from our linear forward operator
rxLoc = survey.srcField.rxList[0].locs
wr = np.zeros(3*nC)
for ii in range(survey.nD):
    wr += ((prob.F[ii, :]*prob.mapping.deriv(m0))/survey.std[ii])**2.

wr = (wr/np.max(wr))
wr = wr**0.5

#Mesh.TensorMesh.writeModelUBC(mesh, work_dir + out_dir + 'SensWeights.sus',
#                              actvMap*(homogMap.P*wr))
# wr = PF.Magnetics.get_dist_wgt(mesh, rxLoc, actv, 3, 1)

# Create a block diagonal regularization
wires = Maps.Wires(('p', nC), ('s', nC), ('t', nC))

idenMap = Maps.IdentityMap(nP=nC)

regMesh = Mesh.TensorMesh([nC])

# Create a regularization
reg_p = Regularization.Sparse(regMesh, mapping=wires.p)
reg_p.cell_weights = wires.p * wr
reg_p.norms = [2, 2, 2, 2]

reg_s = Regularization.Sparse(regMesh, mapping=wires.s)
reg_s.cell_weights = wires.s * wr
reg_s.norms = [2, 2, 2, 2]

reg_t = Regularization.Sparse(regMesh, mapping=wires.t)
reg_t.cell_weights = wires.t * wr
reg_t.norms = [2, 2, 2, 2]

reg = reg_p + reg_s + reg_t
reg.mref = np.zeros(3*nC)

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=10, lower=-10., upper=10.,
                                 maxIterCG=20, tolCG=1e-3)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=1e+5)
betaest = Directives.BetaEstimate_ByEig()

# Here is where the norms are applied
IRLS = Directives.Update_IRLS(f_min_change=1e-4,
                              minGNiter=3, beta_tol=1e-2)

update_Jacobi = Directives.UpdateJacobiPrecond()
targetMisfit = Directives.TargetMisfit()

saveModel = Directives.SaveUBCModelEveryIteration(mapping=actvMap)
saveModel.fileName = work_dir + out_dir + 'MVI_C'
inv = Inversion.BaseInversion(invProb,
                              directiveList=[IRLS, update_Jacobi,
                                             saveModel])

m0 = np.ones(3*nC)*1e-6

mrec_MVI = inv.run(m0)

beta = invProb.beta

# %% RUN MVI-S

# # STEP 3: Finish inversion with spherical formulation
mstart = PF.Magnetics.xyz2atp(mrec_MVI)
prob.coordinate_system = 'spherical'
prob.model = mstart

# Create a block diagonal regularization
wires = Maps.Wires(('amp', nC), ('theta', nC), ('phi', nC))

# Create a regularization
reg_a = Regularization.Sparse(regMesh, mapping=wires.amp)
reg_a.norms = driver.lpnorms[:4]
if driver.eps is not None:
    reg_a.eps_p = driver.eps[0]
    reg_a.eps_q = driver.eps[1]
else:
    reg_a.eps_p = np.percentile(np.abs(mstart[:nC]), 95)

reg_t = Regularization.Sparse(regMesh, mapping=wires.theta)
reg_t.alpha_s = 0.
reg_t.space = 'spherical'
reg_t.norms = driver.lpnorms[4:8]
reg_t.eps_q = 2e-2
# reg_t.alpha_x, reg_t.alpha_y, reg_t.alpha_z = 0.25, 0.25, 0.25

reg_p = Regularization.Sparse(regMesh, mapping=wires.phi)
reg_p.alpha_s = 0.
reg_p.space = 'spherical'
reg_p.norms = driver.lpnorms[8:]
reg_p.eps_q = 1e-2

reg = reg_a + reg_t + reg_p
reg.mref = np.zeros(3*nC)

# Data misfit function
dmis = DataMisfit.l2_DataMisfit(survey)
dmis.W = 1./survey.std

# Add directives to the inversion
opt = Optimization.ProjectedGNCG(maxIter=40,
                                 lower=np.r_[np.zeros(nC), -np.inf*np.ones(nC), -np.inf*np.ones(nC)],
                                 upper=np.r_[np.ones(nC)*10, np.inf*np.ones(nC), np.inf*np.ones(nC)],
                                 maxIterLS=10,
                                 maxIterCG=20, tolCG=1e-3,
                                 stepOffBoundsFact=1e-8, LSalwaysPass=True)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=beta*100.)
#  betaest = Directives.BetaEstimate_ByEig()

# Here is where the norms are applied
IRLS = Directives.Update_IRLS(f_min_change=1e-4,
                              minGNiter=2, beta_tol=1e-2,
                              coolingRate=2, maxIRLSiter=20)

invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=beta)
# betaest = Directives.BetaEstimate_ByEig()

# Special directive specific to the mag amplitude problem. The sensitivity
# weights are update between each iteration.
update_SensWeight = Directives.UpdateSensWeighting()
update_Jacobi = Directives.UpdateJacobiPrecond()
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
PF.MagneticsDriver.writeVectorUBC(mesh,work_dir + out_dir + "MVIS_lp" +'.fld',np.c_[mx, my, mz])

mrec_MVI_S_l2 = PF.Magnetics.atp2xyz(invProb.l2model).reshape(nC,3,order='F')
mx = actvMap*mrec_MVI_S_l2[:,0]
my = actvMap*mrec_MVI_S_l2[:,1]
mz = actvMap*mrec_MVI_S_l2[:,2]
PF.MagneticsDriver.writeVectorUBC(mesh,work_dir + out_dir + "MVIS_l2" + '.fld',np.c_[mx, my, mz])

mrec_MVI = mrec_MVI.reshape((nC, 3), order='F')
mx = actvMap*mrec_MVI[:,0]
my = actvMap*mrec_MVI[:,1]
mz = actvMap*mrec_MVI[:,2]
PF.MagneticsDriver.writeVectorUBC(mesh,work_dir + out_dir + "MVIC" +'.fld',np.c_[mx, my, mz])

# Outputs
PF.Magnetics.writeUBCobs(work_dir+out_dir + "MVIS" +"_Inv.pre", survey, invProb.dpred)
