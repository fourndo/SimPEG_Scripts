from SimPEG import Mesh, Directives, Maps, InvProblem, Optimization, DataMisfit, Inversion, Utils, Regularization
import SimPEG.PF as PF
import numpy as np
import matplotlib.pyplot as plt
import os

#work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Modelling\\Synthetic\\Triple_Block_lined\\"
# work_dir = "C:\\Users\\DominiqueFournier\\ownCloud\\Research\\Synthetic\\Nut_Cracker\\"
work_dir = "C:\\Users\\DominiqueFournier\\Downloads\\Mages_01\\Mages_01\\"
out_dir = "SimPEG_FWR\\"
input_file = "SimPEG_MAG.inp"
# %%
# Read in the input file which included all parameters at once
# (mesh, topo, model, survey, inv param, etc.)
driver = PF.MagneticsDriver.MagneticsDriver_Inv(work_dir + input_file)

os.system('if not exist ' + work_dir + out_dir + ' mkdir ' + work_dir+out_dir)

# Access the mesh and survey information
mesh = driver.mesh
survey = driver.survey

# Get the active cells from model
m0 = driver.m0

print(m0)
actv = m0 != -100

# Get the layer of cells directyl below topo
#surf = Utils.actIndFull2layer(mesh, active)
nC = int(np.sum(actv))  # Number of active cells


# Create identity map
idenMap = Maps.IdentityMap(nP=nC)

# Create static map
prob = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap, actInd=actv)

# Pair the survey and problem
survey.pair(prob)


# %% STEP 2: COMPUTE AMPLITUDE DATA
# Now that we have an equialent source layer, we can forward model alh three
# components of the field and add them up: |B| = ( Bx**2 + Bx**2 + Bx**2 )**0.5

# Won't store the sensitivity and output 'xyz' data.
prob.forwardOnly = True
pred_x = prob.Intrgl_Fwr_Op(m=m0[actv], recType='tmi')

# Form data object with coordinates and write to file
wd = survey.std

# Save forward data to file
PF.Magnetics.writeUBCobs(work_dir + out_dir + 'Predicted_data.obs', survey, pred_x)


