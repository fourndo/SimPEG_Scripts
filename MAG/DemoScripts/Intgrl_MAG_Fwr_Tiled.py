import os

#home_dir = 'C:\\LC\\Private\\dominiquef\\Projects\\4414_Minsim\\Modeling\\MAG\\Composite'
home_dir = 'C:\\Users\\dominiquef.MIRAGEOSCIENCE\\ownCloud\\Research\\Modelling\\Synthetic\\Block_Gaussian_topo'


inpfile = 'PYMAG3C_fwr.inp'

dsep = '\\'

os.chdir(home_dir)

ndv = -100

max_mcell = 1000
min_Olap = 200

#%%
from SimPEG import np, sp, Utils, mkvc, Maps, Mesh
import simpegPF as PF
import pylab as plt

## New scripts to be added to basecode
#from fwr_MAG_data import fwr_MAG_data
#from read_MAGfwr_inp import read_MAGfwr_inp

#%%
# Read input file
[mshfile, obsfile, modfile, magfile, topofile] = PF.Magnetics.read_MAGfwr_inp(inpfile)

# Load mesh file
mesh = Mesh.TensorMesh.readUBC(mshfile)

# Load model file
model = Mesh.TensorMesh.readModelUBC(mesh,modfile)
model[model==ndv] = 0.
  
# Load in topofile or create flat surface
if topofile == 'null':
 
    actv = np.ones(mesh.nC)   
    
else: 
    topo = np.genfromtxt(topofile,skip_header=1)
    actv = PF.Magnetics.getActiveTopo(mesh,topo,'N')


# Create active map to go from reduce set to full
actvMap = Maps.ActiveCells(mesh, actv, -100)

Mesh.TensorMesh.writeModelUBC(mesh,'nullcell.dat',actvMap*np.ones(len(actv)))
         
# Load in observation file
survey = PF.Magnetics.readUBCmagObs(obsfile)

rxLoc = survey.srcField.rxList[0].locs
d = None

#rxLoc[:,2] += 5 # Temporary change for test
ndata = rxLoc.shape[0]

#%% Subdivide the forward data locations into tiles
PF.Magnetics.plot_obs_2D(rxLoc)

xmin = np.min(rxLoc[:,0])
xmax = np.max(rxLoc[:,0])

ymin = np.min(rxLoc[:,1])
ymax = np.max(rxLoc[:,1])

var = np.c_[np.r_[xmin,ymin],np.r_[xmax,ymin],np.r_[xmin,ymax]]
var = np.c_[var.T,np.ones(3)*mesh.vectorCCz[-1]]

# First put a tile on both corners of the mesh
indx = Utils.closestPoints(mesh, var )
endl = np.c_[mesh.gridCC[indx,0],mesh.gridCC[indx,1]]

dx = np.median(mesh.hx)

# Add intermediate tiles until the min_Olap is respected
# First in the x-direction
lx = np.floor( ( max_mcell/ mesh.nCz )**0.5 );
ntile = 2
Olap  = -1

while Olap < min_Olap:
    
    ntile += 1
    
    # Set location of SW corners
    #x0 = np.c_[xmin,xmax-lx*dx]
    
    dx_t = np.round( ( endl[1,0] - endl[0,0] ) / ( (ntile-1) * dx) ) 

    x1 = np.r_[endl[0,0],endl[0,0] + np.cumsum((np.ones(ntile-2)) * dx_t * dx),endl[1,0]]
    x2 = x1 + lx*dx;

    y1 = np.ones(len(x1)) * ymin
    y2 = np.ones(len(x1)) * (ymax + lx * dx)
    
    Olap = x1[0] + lx*dx - x1[1]


#%% Run forward modeling
# Compute forward model using integral equation
#d = PF.Magnetics.Intgrl_Fwr_Data(mesh,B,M,rxLoc,model,actv,'tmi')

# Form data object with coordinates and write to file
#wd =  np.zeros((ndata,1))
#
# Save forward data to file
#PF.Magnetics.writeUBCobs(home_dir + dsep + 'FWR_data.dat',B,M,rxLoc,d,wd)


