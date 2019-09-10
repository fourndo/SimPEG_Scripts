"""
        Experimental script for the forward modeling of DC resistivity data
        along survey lines defined by the user. The program loads in a 3D mesh
        and model which is used to design pole-dipole or dipole-dipole survey
        lines.

        Uses SimPEG to generate the forward problem and compute the LU
        factorization.

        Calls DCIP2D for the inversion of a projected 2D section from the full
        3D model.

        Assumes flat topo for now...

        Created on Mon December 7th, 2015

        @author: dominiquef

"""


#%%
from SimPEG import *
import SimPEG.DCIP as DC
import pylab as plt
from pylab import get_current_fig_manager
from scipy.interpolate import griddata
import time
import re
import numpy.matlib as npm
import scipy.interpolate as interpolation
from matplotlib import animation
from JSAnimation import HTMLWriter

home_dir = 'C:\\Users\\dominiquef.MIRAGEOSCIENCE\\ownCloud\\Research\\MtIsa\\Modeling'
#home_dir = 'C:\\Users\\dominiquef.MIRAGEOSCIENCE\\ownCloud\\Research\\Modelling\\Synthetic\\Two_Sphere'
dsep = '\\'
#from scipy.linalg import solve_banded

# Load UBC mesh 3D
#mesh = Mesh.TensorMesh.readUBC(home_dir + '\Mesh_5m.msh')
mesh = Mesh.TensorMesh.readUBC(home_dir + '\\MtIsa_20m.msh')
#mesh = Utils.meshutils.readUBCTensorMesh(home_dir + '\Mesh_50m.msh')

# Load model
model = Mesh.TensorMesh.readModelUBC(mesh,home_dir + '\\MtIsa_20m.con')
#model = Utils.meshutils.readUBCTensorModel(home_dir + '\Synthetic.con',mesh)
#model = Utils.meshutils.readUBCTensorModel(home_dir + '\Lalor_model_50m.con',mesh)
#model = Mesh.TensorMesh.readModelUBC(mesh,home_dir + '\TwoSpheres.con')

#model = model**0 * 1e-2
# Specify survey type
stype = 'pdp'
dtype = 'appc'

# Survey parameters
a = 20
b = 10
n = 1

# Forward solver
slvr = 'BiCGStab' #'LU'

# Preconditioner
pcdr = 'Jacobi'#

# Inversion parameter
pct = 0.01
flr = 5e-5
chifact = 100
ref_mod = 1e-2

# DOI threshold
cutoff = 0.8

# number of padding cells
padc = 0

# Plotting param
#xmin, xmax = -200, 200
#zmin, zmax = -200, 0
xmin, xmax = 11300, 12700
zmin, zmax = -80, 520
vmin = -2.
vmax = 1.
depth = 600. # Maximum depth to plot
dx_in = 10.
#%% Create system
#Set boundary conditions
mesh.setCellGradBC('neumann')

Div = mesh.faceDiv
Grad = mesh.cellGrad
Msig = Utils.sdiag(1./(mesh.aveF2CC.T*(1./model)))

A = Div*Msig*Grad

# Change one corner to deal with nullspace
A[0,0] = 1
A = sp.csc_matrix(A)

start_time = time.time()

if re.match(slvr,'BiCGStab'):
    # Create Jacobi Preconditioner
    if re.match(pcdr,'Jacobi'):
        dA = A.diagonal()
        P = sp.spdiags(1/dA,0,A.shape[0],A.shape[0])

        #LDinv = sp.linalg.splu(LD)

elif re.match(slvr,'LU'):
    # Factor A matrix
    Ainv = sp.linalg.splu(A)
    print("LU DECOMP--- %s seconds ---" % (time.time() - start_time))

#%% Create survey
# Display top section
top = int(mesh.nCz)-1

plt.figure()
ax_prim = plt.subplot(1,1,1)
dat1 = mesh.plotSlice(model, ind=top, normal='Z', grid=False, pcolorOpts={'alpha':0.5}, ax =ax_prim)
#==============================================================================
# plt.xlim([423200,423750])
# plt.ylim([546350,546650])
#==============================================================================
plt.gca().set_aspect('equal', adjustable='box')

plt.show()
cfm1=get_current_fig_manager().window
gin=[1]

# Keep creating sections until returns an empty ginput (press enter on figure)
#while bool(gin)==True:

# Bring back the plan view figure and pick points
cfm1.activateWindow()
plt.sca(ax_prim)

# Takes two points from ginput and create survey
#if re.match(stype,'gradient'):
#gin = [(-200.  ,  0.), (200.  ,  0.)]
#else:
#gin = plt.ginput(2, timeout = 0)
gin = [(11350, 12200), (12650, 12200)]


#==============================================================================
# if not gin:
#     print 'SimPED - Simulation has ended with return'
#     break
#==============================================================================

# Add z coordinate to all survey... assume flat
nz = mesh.vectorNz
var = np.c_[np.asarray(gin),np.ones(2).T*nz[-1]]

# Snap the endpoints to the grid. Easier to create 2D section.
indx = Utils.closestPoints(mesh, var )
endl = np.c_[mesh.gridCC[indx,0],mesh.gridCC[indx,1],np.ones(2).T*nz[-1]]

[survey2D, Tx, Rx] = DC.gen_DCIPsurvey(endl, mesh, stype, a, b, n)

dl_len = np.sqrt( np.sum((endl[0,:] - endl[1,:])**2) )
dl_x = ( Tx[-1][0,1] - Tx[0][0,0] ) / dl_len
dl_y = ( Tx[-1][1,1] - Tx[0][1,0]  ) / dl_len
azm =  np.arctan(dl_y/dl_x)

#%% Create a 2D mesh along axis of Tx end points and keep z-discretization
dx = np.min( [ np.min(mesh.hx), np.min(mesh.hy), dx_in ])
ncx = np.ceil(dl_len/dx)+3
ncz = np.ceil( depth / dx )

padx = dx*np.power(1.4,range(1,padc))

# Creating padding cells
hx = np.r_[padx[::-1], np.ones(ncx)*dx , padx]
hz = np.r_[padx[::-1], np.ones(ncz)*dx]

# Create 2D mesh
x0 = gin[0][0] - np.sum(padx) * np.cos(azm)
y0 = gin[0][1] - np.sum(padx) * np.sin(azm)

# Create mesh one cell below the 3D mesh to avoid the source 
mesh2d = Mesh.TensorMesh([hx, hz], x0=(x0,mesh.vectorNz[-2] - np.sum(hz) ))



#%% Create array of points for interpolating from 3D to 2D mesh
xx = x0 + (np.cumsum(mesh2d.hx) - mesh2d.hx/2) * np.cos(azm)
yy = y0 + (np.cumsum(mesh2d.hx) - mesh2d.hx/2) * np.sin(azm)
zz = mesh2d.vectorCCy

[XX,ZZ] = np.meshgrid(xx,zz)
[YY,ZZ] = np.meshgrid(yy,zz)

xyz2d = np.c_[mkvc(XX),mkvc(YY),mkvc(ZZ)]

#plt.scatter(xx,yy,s=20,c='y')


F = interpolation.LinearNDInterpolator(mesh.gridCC,model)
m2D = np.reshape(F(xyz2d),[mesh2d.nCx,mesh2d.nCy]).T

#%% Forward model data
fig, axs = plt.subplots(1,1, figsize = (6,4))

plt.tight_layout(pad=0.5)

#pos =  axs.get_position() 
#axs.set_position([pos.x0 , pos.y0+0.2,  pos.width, pos.height])
    
#im1 = axs.pcolormesh([],[],[], alpha=0.75,extent = (xx[0],xx[-1],yy[-1],yy[0]),interpolation='nearest',vmin=-1e-2, vmax=1e-2)
#im2 = axs.pcolormesh([],[],[],alpha=0.2,extent = (xx[0],xx[-1],yy[-1],yy[0]),interpolation='nearest',cmap='gray')
#im1 = axs.pcolormesh(xx,zz,np.zeros((mesh2d.nCy,mesh2d.nCx)), alpha=0.75,vmin=-1e-2, vmax=1e-2)
im2 = axs.pcolormesh(xx,zz,np.zeros((mesh2d.nCy,mesh2d.nCx)), alpha=0.75,vmin=-1e-2, vmax=1e-2)
cbar = plt.colorbar(im2,format="$10^{%.1f}$",fraction=0.04,orientation="horizontal")
im3 = axs.streamplot(xx, zz, np.zeros((mesh2d.nCy,mesh2d.nCx)), np.zeros((mesh2d.nCy,mesh2d.nCx)),color='k')
im4 = axs.scatter([],[], c='r', s=200)
im5 = axs.scatter([],[], c='r', s=200)


problem = DC.ProblemDC_CC(mesh)
tinf = np.squeeze(Rx[-1][-1,:3]) + np.array([dl_x,dl_y,0])*10*a
def animate(ii):
    
    
    removeStream()
    
    
    
    if not re.match(stype,'pdp'):
        
        inds = Utils.closestPoints(mesh, np.asarray(Tx[ii]).T )
        RHS = mesh.getInterpolationMat(np.asarray(Tx[ii]).T, 'CC').T*( [-1,1] / mesh.vol[inds] )
    
    else:
    
        # Create an "inifinity" pole
        tx =  np.squeeze(Tx[ii][:,0:1])
        #tinf = tx + np.array([dl_x,dl_y,0])*dl_len
        inds = Utils.closestPoints(mesh, np.c_[tx,tinf].T)
        RHS = mesh.getInterpolationMat(np.c_[tx,tinf].T, 'CC').T*( [-1,1] / mesh.vol[inds] )
    
    
    if re.match(slvr,'BiCGStab'):
    
        if re.match(pcdr,'Jacobi'):
            dA = A.diagonal()
            P = sp.spdiags(1/dA,0,A.shape[0],A.shape[0])
    
            # Iterative Solve
            Ainvb = sp.linalg.bicgstab(P*A,P*RHS, tol=1e-5)
    
    
        phi = mkvc(Ainvb[0])
    
    elif re.match(slvr,'LU'):
        #Direct Solve
        phi = Ainv.solve(RHS)
    
    
    j = -Msig*Grad*phi
    j_CC = mesh.aveF2CCV*j
    
    # Compute charge density solving div*grad*phi
    Q = -mesh.faceDiv*mesh.cellGrad*phi
    
    jx_CC = j_CC[0:mesh.nC]
    jy_CC = j_CC[(2.*mesh.nC):]
    
    #%% Grab only the core for presentation
    F = interpolation.NearestNDInterpolator(mesh.gridCC,jx_CC)   
    jx_CC_sub =  np.reshape(F(xyz2d),[mesh2d.nCx,mesh2d.nCy]).T
    
    F = interpolation.NearestNDInterpolator(mesh.gridCC,jy_CC)   
    jy_CC_sub =  np.reshape(F(xyz2d),[mesh2d.nCx,mesh2d.nCy]).T
    
    F = interpolation.NearestNDInterpolator(mesh.gridCC,Q) 
    Q_sub = np.reshape(F(xyz2d),[mesh2d.nCx,mesh2d.nCy]).T
    
    J_rho = np.sqrt(jx_CC_sub**2 + jy_CC_sub**2)
    lw = np.log10(J_rho/J_rho.min())
    
    # Normalize the charge density from -1 to 1
    Q_sub[Q_sub<0] = Q_sub[Q_sub<0]/np.abs(np.min(Q_sub[Q_sub<0]))
    Q_sub[Q_sub>0] = Q_sub[Q_sub>0]/np.abs(np.max(Q_sub[Q_sub>0]))    
    
    #axs.imshow(Q_sub,alpha=0.75,extent = (xx[0],xx[-1],yy[-1],yy[0]),interpolation='nearest',vmin=-1e-2, vmax=1e-2)    
    #axs.imshow(np.log10(model_sub.reshape(mesh2d.nCy,mesh2d.nCx)),alpha=0.2,extent = (xx[0],xx[-1],yy[-1],yy[0]),interpolation='nearest',cmap='gray')     
    #global im1
    #im1 = axs.pcolormesh(mesh2d.vectorCCx,mesh2d.vectorCCy,Q_sub, alpha=0.75,vmin=-1e-4, vmax=1e-4)
    
    global im2, cbar
    axs.pcolormesh(mesh2d.vectorCCx,mesh2d.vectorCCy,np.log10(m2D), alpha=0.25, cmap = 'gray')
    im2 = axs.pcolormesh(mesh2d.vectorCCx,mesh2d.vectorCCy,Q_sub, alpha=0.25, vmin=-1,vmax = 1, cmap = 'bwr')
    
    # Add colorbar
    cbar = fig.colorbar(im2, orientation="horizontal",ticks=np.linspace(-1,1, 3))
    cbar.set_label("Normalized Charge Density",size=10)
    
    
    global im3
    im3 = axs.streamplot(xx, zz, jx_CC_sub/J_rho.max(), jy_CC_sub/J_rho.max(),color='k',density=0.5, linewidth = lw)
    
    global im4
    im4 = axs.scatter(Tx[ii][0,0],Tx[ii][2,0], c='r', s=100, marker='v' )
    
    global im5
    im5 = axs.scatter(Tx[ii][0,1],Tx[ii][2,1], c='b', s=100, marker='v' )
        
    plt.ylim(zz[0],zz[-1]+3*dx)
    plt.xlim(xx[0]-dx,xx[-1]+dx)
    plt.gca().set_aspect('equal', adjustable='box')
    x = np.linspace(xmin,xmax, 5)
    axs.set_xticks(map(int, x))
    axs.set_xticklabels(map(str, map(int, x)),size=12)
    z = np.linspace(zmin,zmax, 5)
    axs.set_yticks(map(int, z))
    axs.set_yticklabels(map(str, map(int, z)),size=12)
    #axs.set_title("Conductivity (S/m) and Current")
    #plt.show()
    #im1.set_array(Q_sub)
    #im2.set_array(np.log10(model_sub.reshape(mesh2d.nCy,mesh2d.nCx)))
    #im2.set_array(mesh2d.vectorCCx, mesh2d.vectorCCy,jx_CC_sub.T,jy_CC_sub.T)

    #return [im1] + [im2]
#%% Create widget
def removeStream():
    #global im1
    #im1.remove()    
    
    global im2, cbar
    im2.remove()  
    cbar.remove()
    global im3
    im3.lines.remove()
    axs.patches = []
    
    global im4
    im4.remove()
    
    global im5
    im5.remove()
#def viewInv(msh,iteration):



#, linewidth=lw.T
#%%   
#interact(viewInv,msh = mesh2d, iteration = IntSlider(min=0, max=len(txii)-1 ,step=1, value=0))
# set embed_frames=True to embed base64-encoded frames directly in the HTML
anim = animation.FuncAnimation(fig, animate,
                               frames=survey2D.nSrc , interval=500)
                               #
anim.save(home_dir + '\\animation.html', writer=HTMLWriter(embed_frames=True,fps=1))
