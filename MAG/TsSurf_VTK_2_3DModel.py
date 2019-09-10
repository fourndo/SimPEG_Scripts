import os

home_dir = 'C:\\LC\\Private\\dominiquef\\Projects\\4414_Minsim\\Modeling\\MAG\\Lalor'

inpfile = 'PYMAG3C_fwr.inp'

dsep = '\\'

os.chdir(home_dir)

#%%
from SimPEG import *
from SimPEG.Utils import io_utils
import time as tm
import vtk, vtk.util.numpy_support as npsup
import re

"""
Loads in a triangulated surface from Gocad (*.ts) and use VTK to transfer onto
a 3D mesh.

New scripts to be added to basecode
"""
#%%

work_dir = 'C:\\Users\\dominiquef.MIRAGEOSCIENCE\\Google Drive\\Tli_Kwi_Cho\\Modelling\\Geological_model'

mshfile = '\\MEsh_TEst.msh'

# Load mesh file
mesh = Mesh.TensorMesh.readUBC(work_dir+mshfile)

# Load in observation file
#[B,M,dobs] = PF.BaseMag.readUBCmagObs(obsfile)

# Read in topo surface
topsurf = work_dir+'\\CDED_Lake_Coarse.ts'
geosurf = [[work_dir+'\\XVK.ts',True,True]]
#            [work_dir+'\\PK2.ts',True,True],
#           [work_dir+'\\PK3.ts',True,True],
#[work_dir+'\\HK1.ts',True,True],
#[work_dir+'\\VK.ts',True,True]
#]

# Background density
bkgr = 1e-4
airc = 1e-8

# Units
vals = np.asarray([5e-2,1e-2,1e-2,1e-3,5e-3])




#%% Script starts here       
# # Create a grid of observations and offset the z from topo

model= np.ones(mesh.nC) * bkgr
# Load GOCAD surf
#[vrtx, trgl] = PF.BaseMag.read_GOCAD_ts(tsfile)
# Find active cells from surface

modelInd = np.ones(mesh.nC)*ndv
for ii in range(len(geosurf)):
    tin = tm.time()
    print "Computing indices with VTK: " + geosurf[ii][0]
    T, S = io_utils.read_GOCAD_ts(geosurf[ii][0])
    indx = io_utils.surface2inds(T,S,mesh, boundaries=geosurf[ii][1], internal=geosurf[ii][2])
    print "VTK operation completed in " + str(tm.time() - tin) + " sec"
    modelInd[indx] = geosurf[ii][3]
    
def getModel(R1=1e-2, R2=1e-1):
    vals = [R1, R2]
    model= np.ones(mesh.nC) * bkgr

    for ii, sus in zip(range(7),vals):
        model[modelInd == ii] = sus
    return model
    
model = getModel()