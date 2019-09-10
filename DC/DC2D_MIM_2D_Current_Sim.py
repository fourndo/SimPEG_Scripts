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
indmin = model>1.25
model[indmin] = 10

model2 = model*1.
model2[indmin] = 0.08

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
Msig1 = Utils.sdiag(1./(mesh.aveF2CC.T*(1./model)))
Msig2 = Utils.sdiag(1./(mesh.aveF2CC.T*(1./model2)))



        #LDinv = sp.linalg.splu(LD)

#==============================================================================
# elif re.match(slvr,'LU'):
#     # Factor A matrix
#     Ainv = sp.linalg.splu(A)
#     print("LU DECOMP--- %s seconds ---" % (time.time() - start_time))
#==============================================================================

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


F = interpolation.NearestNDInterpolator(mesh.gridCC,model)
m2D = np.reshape(F(xyz2d),[mesh2d.nCx,mesh2d.nCy]).T

F = interpolation.NearestNDInterpolator(mesh.gridCC,model2)
m2D2 = np.reshape(F(xyz2d),[mesh2d.nCx,mesh2d.nCy]).T

#%% Forward model data
#fig, axs = plt.subplots(2,1, figsize = (6,4))

fig = plt.figure(figsize=(7,7))
ax1 = fig.add_subplot(2,1,2, aspect='equal')
ax2 = fig.add_subplot(2,1,1, aspect='equal')

pos =  ax1.get_position()
ax1.set_position([pos.x0, pos.y0,  pos.width, pos.height])
#==============================================================================
# if pp == 1:
#     pos =  ax.get_position()
#     ax.set_position([pos.x0, pos.y0+.05,  pos.width, pos.height])
#==============================================================================

#plt.tight_layout(pad=0.5)

#pos =  axs.get_position()
#axs.set_position([pos.x0 , pos.y0+0.2,  pos.width, pos.height])

#im1 = axs.pcolormesh([],[],[], alpha=0.75,extent = (xx[0],xx[-1],yy[-1],yy[0]),interpolation='nearest',vmin=-1e-2, vmax=1e-2)
#im2 = axs.pcolormesh([],[],[],alpha=0.2,extent = (xx[0],xx[-1],yy[-1],yy[0]),interpolation='nearest',cmap='gray')
#im1 = axs.pcolormesh(xx,zz,np.zeros((mesh2d.nCy,mesh2d.nCx)), alpha=0.75,vmin=-1e-2, vmax=1e-2)
im2 = ax1.pcolormesh(xx,zz,np.zeros((mesh2d.nCy,mesh2d.nCx)), alpha=0.75,vmin=-2, vmax=1)
im3 = ax1.streamplot(xx, zz, np.zeros((mesh2d.nCy,mesh2d.nCx)), np.zeros((mesh2d.nCy,mesh2d.nCx)),color='k')
im4 = ax1.scatter([],[], c='r', s=200)
im5 = ax1.scatter([],[], c='r', s=200)

im6 = ax2.pcolormesh(xx,zz,np.zeros((mesh2d.nCy,mesh2d.nCx)), alpha=0.75,vmin=-2, vmax=1)
im7 = ax2.streamplot(xx, zz, np.zeros((mesh2d.nCy,mesh2d.nCx)), np.zeros((mesh2d.nCy,mesh2d.nCx)),color='k')
im8 = ax2.scatter([],[], c='r', s=200)
im9 = ax2.scatter([],[], c='r', s=200)

pos =  ax1.get_position()
cbarax = fig.add_axes([pos.x0+0.2, pos.y0-.04,  pos.width*0.5, pos.height*0.05])  ## the parameters are the specified position you set
cb = plt.colorbar(im2, cax=cbarax, orientation="horizontal", ax = ax1, ticks=np.linspace(-2,1, 3))
cb.set_label("Conductivity log(S/m)",size=10)

problem = DC.ProblemDC_CC(mesh)
tinf = np.squeeze(Rx[-1][-1,:3]) + np.array([dl_x,dl_y,0])*10*a
def animate(ii):


    removeStream()

    global ax1, ax2, fig


    for pp in range(2):

        if pp == 0:

            Msig = Msig1

        else:

            Msig = Msig2

        A = Div*Msig*Grad

        # Change one corner to deal with nullspace
        A[0,0] = 1
        A = sp.csc_matrix(A)


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
        jy_CC = j_CC[(2*mesh.nC):]

        #%% Grab only the core for presentation
        F = interpolation.NearestNDInterpolator(mesh.gridCC,jx_CC)
        jx_CC_sub =  np.reshape(F(xyz2d),[mesh2d.nCx,mesh2d.nCy]).T

        F = interpolation.NearestNDInterpolator(mesh.gridCC,jy_CC)
        jy_CC_sub =  np.reshape(F(xyz2d),[mesh2d.nCx,mesh2d.nCy]).T

        F = interpolation.NearestNDInterpolator(mesh.gridCC,Q)
        Q_sub = np.reshape(F(xyz2d),[mesh2d.nCx,mesh2d.nCy]).T

        J_rho = np.sqrt(jx_CC_sub**2 + jy_CC_sub**2)
        #lw = np.log10(J_rho/J_rho.min())

        # Normalize the charge density from -1 to 1
        Q_sub[Q_sub<0] = Q_sub[Q_sub<0]/np.abs(np.min(Q_sub[Q_sub<0]))
        Q_sub[Q_sub>0] = Q_sub[Q_sub>0]/np.abs(np.max(Q_sub[Q_sub>0]))

        #axs.imshow(Q_sub,alpha=0.75,extent = (xx[0],xx[-1],yy[-1],yy[0]),interpolation='nearest',vmin=-1e-2, vmax=1e-2)
        #axs.imshow(np.log10(model_sub.reshape(mesh2d.nCy,mesh2d.nCx)),alpha=0.2,extent = (xx[0],xx[-1],yy[-1],yy[0]),interpolation='nearest',cmap='gray')
        #global im1
        #im1 = axs.pcolormesh(mesh2d.vectorCCx,mesh2d.vectorCCy,Q_sub, alpha=0.75,vmin=-1e-4, vmax=1e-4)


        if pp == 0:

            ax1 = fig.add_subplot(2,1,2, aspect='equal')


            im2 = ax1.pcolormesh(mesh2d.vectorCCx,mesh2d.vectorCCy,np.log10(m2D), vmin=-2,vmax=1)
            #ax1.pcolormesh(mesh2d.vectorCCx,mesh2d.vectorCCy,Q_sub, alpha=0.25, vmin=-1,vmax = 1, cmap = 'bwr')


            im3 = ax1.streamplot(xx, zz, jx_CC_sub/J_rho.max(), jy_CC_sub/J_rho.max(),color='w',density=0.5, linewidth = 2)


            im4 = ax1.scatter(Tx[ii][0,0],Tx[ii][2,0], c='r', s=100, marker='v' )


            im5 = ax1.scatter(Tx[ii][0,1],Tx[ii][2,1], c='b', s=100, marker='v' )


            plt.ylim(zz[0],zz[-1]+3*dx)
            plt.xlim(xx[0]-dx,xx[-1]+dx)
            plt.gca().set_aspect('equal', adjustable='box')
            x = np.linspace(xmin,xmax, 5)
            ax1.set_xticks(map(int, x))
            ax1.set_xticklabels(map(str, map(int, x)),size=12)
            z = np.linspace(zmin,zmax, 5)
            ax1.set_yticks(map(int, z))
            ax1.set_yticklabels(map(str, map(int, z)),size=12)
            ax1.xaxis.tick_top()




        else:

            ax2 = fig.add_subplot(2,1,1, aspect='equal')
            im6 = ax2.pcolormesh(mesh2d.vectorCCx,mesh2d.vectorCCy,np.log10(m2D2), vmin=-2,vmax=1)
            #ax2.pcolormesh(mesh2d.vectorCCx,mesh2d.vectorCCy,Q_sub, alpha=0.25, vmin=-1,vmax = 1, cmap = 'bwr')
    #==============================================================================
    #             cbar = fig.colorbar(im6, orientation="horizontal",ticks=np.linspace(-1,1, 3))
    #             cbar.set_label("Normalized Charge Density",size=10)
    #==============================================================================


            im7 = ax2.streamplot(xx, zz, jx_CC_sub/J_rho.max(), jy_CC_sub/J_rho.max(),color='w',density=0.5, linewidth = 2)

            im8 = ax2.scatter(Tx[ii][0,0],Tx[ii][2,0], c='r', s=100, marker='v' )

            im9 = ax2.scatter(Tx[ii][0,1],Tx[ii][2,1], c='b', s=100, marker='v' )

            plt.ylim(zz[0],zz[-1]+3*dx)
            plt.xlim(xx[0]-dx,xx[-1]+dx)
            plt.gca().set_aspect('equal', adjustable='box')
            x = np.linspace(xmin,xmax, 5)
            ax2.set_xticks(map(int, x))
            ax2.set_xticklabels(map(str, map(int, x)),size=12)
            z = np.linspace(zmin,zmax, 5)
            ax2.set_yticks(map(int, z))
            ax2.set_yticklabels(map(str, map(int, z)),size=12)

            ax2.set_xticklabels([])


#==============================================================================
#     pos =  ax2.get_position()
#     cbarax = fig.add_axes([pos.x0+0.2, pos.y0-.04,  pos.width*0.5, pos.height*0.05])  ## the parameters are the specified position you set
#     cb = plt.colorbar(im2, cax=cbarax, orientation="horizontal", ax = ax2, ticks=np.linspace(-2,1, 3))
#     cb.set_label("Conductivity log(S/m)",size=10)
#==============================================================================
#==============================================================================

#==============================================================================

    #axs.set_title("Conductivity (S/m) and Current")
    #plt.show()
    #im1.set_array(Q_sub)
    #im2.set_array(np.log10(model_sub.reshape(mesh2d.nCy,mesh2d.nCx)))
    #im2.set_array(mesh2d.vectorCCx, mesh2d.vectorCCy,jx_CC_sub.T,jy_CC_sub.T)

    #return [im1] + [im2]
#%% Create widget
def removeStream():
    global ax1, ax2, fig
    #im1.remove()

    fig.delaxes(ax1)
    fig.delaxes(ax2)
    plt.draw()

    #cbarax.patches = []
#==============================================================================
#     global im6
#     im6.remove()
#
#
# #==============================================================================
# #     cbar.remove()
# #==============================================================================
#     global im7
#     im7.lines.remove()
#     ax2.patches = []
#
#     global im8
#     im8.remove()
#
#     global im9
#     im9.remove()
#==============================================================================
#def viewInv(msh,iteration):



#, linewidth=lw.T
#%%
#interact(viewInv,msh = mesh2d, iteration = IntSlider(min=0, max=len(txii)-1 ,step=1, value=0))
# set embed_frames=True to embed base64-encoded frames directly in the HTML
anim = animation.FuncAnimation(fig, animate,
                               frames=survey2D.nSrc , interval=500)
                               ##
anim.save(home_dir + '\\animation.html', writer=HTMLWriter(embed_frames=True,fps=1))
