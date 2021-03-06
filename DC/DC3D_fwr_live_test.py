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
from SimPEG import np, Utils, Mesh, mkvc, sp
import SimPEG.EM.Static.DC as DC
import SimPEG.EM.Static.Utils as DCUtils
import pylab as plt
from pylab import get_current_fig_manager
from scipy.interpolate import griddata
import time
import re
import numpy.matlib as npm

import scipy.interpolate as interpolation

import os

#home_dir = 'C:\\Users\\dominiquef.MIRAGEOSCIENCE\\ownCloud\\Research\\Modelling\\Synthetic\\Two_Sphere'
home_dir ='C:\\Users\\dominiquef.MIRAGEOSCIENCE\\ownCloud\\Research\\MtIsa\\Modeling'
dsep = '\\'
#from scipy.linalg import solve_banded

# Load UBC mesh 3D
#mesh = Utils.meshutils.readUBCTensorMesh(home_dir + '\Mesh_10m.msh')
mesh = Mesh.TensorMesh.readUBC(home_dir + '\MtIsa_20m.msh')
#mesh = Utils.meshutils.readUBCTensorMesh(home_dir + '\Mesh_50m.msh')

# Load model
model = Mesh.TensorMesh.readModelUBC(mesh,home_dir + '\MtIsa_20m.con')
#model = Utils.meshutils.readUBCTensorModel(home_dir + '\Synthetic.con',mesh)
#model = Utils.meshutils.readUBCTensorModel(home_dir + '\Lalor_model_50m.con',mesh)
#model = Utils.meshutils.readUBCTensorModel(home_dir + '\TwoSpheres.con',mesh)

#model = model**0 * 1e-2
# Specify survey type
stype = 'pole-dipole'

# Survey parameters
a = 50
b = 500
n = 40

# Forward solver
slvr = 'BiCGStab' #'LU'

# Preconditioner
pcdr = 'Jacobi'#'Gauss-Seidel'#

# Inversion parameter
pct = 0.01
flr = 1e-4
chifact = 100
ref_mod = 1e-1

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
    
    # Create Gauss-Seidel Preconditioner
    elif re.match(pcdr,'Gauss-Seidel'):
        LD = sp.tril(A,k=0)
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
mesh.plotSlice(model, ind=top, normal='Z', grid=False, pcolorOpts={'alpha':0.5}, ax =ax_prim)
#plt.xlim([423000,424000])
#plt.ylim([546200,547000])
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
#gin = [(425347, 6079766), (427792, 6081806)]
#else:
gin = plt.ginput(2, timeout = 0)



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
      
survey = DCUtils.gen_DCIPsurvey(endl, mesh, stype, a, b, n)
 
#survey = DC.Survey(SrcList)

Tx = DCUtils.getSrc_locs(survey)
Rx = survey.srcList[0].rxList[0].locs

dl_len = np.sqrt( np.sum((endl[0,:] - endl[1,:])**2) ) 
dl_x = ( Tx[-1][3] - Tx[0][0] ) / dl_len
dl_y = ( Tx[-1][4] - Tx[0][1]  ) / dl_len
azm =  np.arctan(dl_y/dl_x)
  
# Plot stations along line   
plt.scatter(Tx[:,[0,3]],Tx[:,[1,4]],s=20,c='g')
plt.scatter(Rx[0][:,0::3],Rx[0][:,1::3],s=20,c='y')

#%% Forward model data
data = []#np.zeros( nstn*nrx )
unct = []
problem = DC.Problem3D_CC(mesh)
    
for ii in range(len(Tx)):
    
    start_time = time.time()
    
    Rx = survey.srcList[ii].rxList[0].locs
    
    # Select dipole locations for receiver
    rxloc_M = np.asarray(Rx[0])
    rxloc_N = np.asarray(Rx[1])
    
    # Number of receivers
    nrx = rxloc_M.shape[0]

    
    
    if not re.match(stype,'pole-dipole'):
        inds = Utils.closestPoints(mesh, np.asarray(Tx[ii].reshape((2,3))) )
        RHS = mesh.getInterpolationMat(np.asarray(Tx[ii].reshape((2,3))), 'CC').T*( [-1,1] / mesh.vol[inds] )   
        
    else: 
        
        # Create an "inifinity" pole
        tx =  np.squeeze(Tx[ii][:,0:1])
        tinf = tx + np.array([dl_x,dl_y,0])*dl_len*2
        inds = Utils.closestPoints(mesh, np.c_[tx,tinf].T)
        RHS = mesh.getInterpolationMat(np.asarray(Tx[ii]).T, 'CC').T*( [-1] / mesh.vol[inds] )  
    
    # Solve for phi on pole locations
    P1 = mesh.getInterpolationMat(rxloc_M, 'CC')
    P2 = mesh.getInterpolationMat(rxloc_N, 'CC')

    if re.match(slvr,'BiCGStab'):
        
        if re.match(pcdr,'Jacobi'):
            dA = A.diagonal()
            P = sp.spdiags(1/dA,0,A.shape[0],A.shape[0])
    
            # Iterative Solve
            Ainvb = sp.linalg.bicgstab(P*A,P*RHS, tol=1e-5)
            
        # Create Gauss-Seidel Preconditioner
        elif re.match(pcdr,'Gauss-Seidel'):
            LD = sp.tril(A,k=0)
            

        phi = mkvc(Ainvb[0])
               
    elif re.match(slvr,'LU'):
        #Direct Solve
        phi = Ainv.solve(RHS)        
    
    
    
    # Compute potential at each electrode
    dtemp = (P1*phi - P2*phi)*np.pi
    
    data = np.hstack([data, dtemp] )     
    unct = np.hstack([unct, np.abs(dtemp) * pct + flr])
   
    print("--- %s seconds ---" % (time.time() - start_time))  
    

#%% Run 2D inversion if pdp or dpdp survey
# Otherwise just plot and apparent susceptibility map

survey.dobs = np.asarray(data)
survey.std = np.asarray(unct)

if not re.match(stype,'gradient'):
    
    #%% Write data file in UBC-DCIP3D format
    DCUtils.writeUBC_DCobs(home_dir+'\FWR_data3D.dat',survey,'3D','GENERAL')     
    
    
    #%% Load 3D data
    DCfile = DCUtils.readUBC_DC3Dobs(home_dir + '\FWR_data3D.dat')
    
    survey = DCfile['DCsurvey']
    
    #%% Convert 3D obs to 2D and write to file
    lineID = DCUtils.xy_2_lineID(survey)
    
    survey2D = DCUtils.convertObs_DC3D_to_2D(survey,lineID)
    
    
    DCUtils.writeUBC_DCobs(home_dir+'\FWR_3D_2_2D.dat',survey2D,'2D','SURFACE')        
    
    #%% Create a 2D mesh along axis of Tx end points and keep z-discretization    
    dx = np.min( [ np.min(mesh.hx), np.min(mesh.hy) ])
    nc = np.ceil(dl_len/dx)+3
    
    padx = dx*np.power(1.4,range(1,15))
    
    # Creating padding cells
    h1 = np.r_[padx[::-1], np.ones(nc)*dx , padx]
    
    # Create mesh with 0 coordinate centerer on the ginput points in cell center
    mesh2d = Mesh.TensorMesh([h1, mesh.hz], x0=(-np.sum(padx)-dx/2,mesh.x0[2]))
    
    # Create array of points for interpolating from 3D to 2D mesh
    xx = Tx[0][0] + mesh2d.vectorCCx * np.cos(azm)
    yy = Tx[0][3] + mesh2d.vectorCCx * np.sin(azm)
    zz = mesh2d.vectorCCy
    
    [XX,ZZ] = np.meshgrid(xx,zz)
    [YY,ZZ] = np.meshgrid(yy,zz)
    
    xyz2d = np.c_[mkvc(XX),mkvc(YY),mkvc(ZZ)]
    
    #plt.scatter(xx,yy,s=20,c='y')
    
    
    F = interpolation.NearestNDInterpolator(mesh.gridCC,model)
    m2D = np.reshape(F(xyz2d),[mesh2d.nCx,mesh2d.nCy]).T
    
    
    #==============================================================================
    # mesh2d = Mesh.TensorMesh([mesh.hx, mesh.hz], x0=(mesh.x0[0]-endl[0,0],mesh.x0[2]))
    # m3D = np.reshape(model, (mesh.nCz, mesh.nCy, mesh.nCx))
    # m2D = m3D[:,1,:]
    #==============================================================================
    
    plt.figure()
    axs = plt.subplot(2,1,1)
    
    plt.xlim([0,nc*dx])
    plt.ylim([mesh2d.vectorNy[-1]-dl_len,mesh2d.vectorNy[-1]])
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.pcolormesh(mesh2d.vectorNx,mesh2d.vectorNy,np.log10(m2D),alpha=0.5, cmap='gray')#axes = [mesh2d.vectorNx[0],mesh2d.vectorNx[-1],mesh2d.vectorNy[0],mesh2d.vectorNy[-1]])
    #mesh2d.plotImage(mkvc(m2D), grid=True, ax=axs)
    
    #%% Plot pseudo section
    ax2 = plt.subplot(2,1,2)
    DCUtils.plot_pseudoSection(survey2D,ax2)
    plt.colorbar
    plt.show()

    #%% Create dcin2d inversion files and run
    inv_dir = home_dir + '\Inv2D' 
    if not os.path.exists(inv_dir):
        os.makedirs(inv_dir)
        
    mshfile2d = 'Mesh_2D.msh'
    modfile2d = 'MtIsa_2D.con'
    obsfile2d = 'FWR_3D_2_2D.dat'
    inp_file = 'dcinv2d.inp'
    
    
    # Export 2D mesh
    fid = open(inv_dir + dsep + mshfile2d,'w')
    fid.write('%i\n'% mesh2d.nCx)
    fid.write('%f %f 1\n'% (mesh2d.vectorNx[0],mesh2d.vectorNx[1]))  
    np.savetxt(fid, np.c_[mesh2d.vectorNx[2:],np.ones(mesh2d.nCx-1)], fmt='\t %e %i',delimiter=' ',newline='\n')
    fid.write('\n')
    fid.write('%i\n'% mesh2d.nCy)
    fid.write('%f %f 1\n'%( 0,mesh2d.hy[-1]))   
    np.savetxt(fid, np.c_[np.cumsum(mesh2d.hy[-2::-1])+mesh2d.hy[-1],np.ones(mesh2d.nCy-1)], fmt='\t %e %i',delimiter=' ',newline='\n')
    fid.close()
    
    # Export 2D model
    fid = open(inv_dir + dsep + modfile2d,'w')
    fid.write('%i %i\n'% (mesh2d.nCx,mesh2d.nCy))
    np.savetxt(fid, mkvc(m2D[::-1,:].T), fmt='%e',delimiter=' ',newline='\n')
    fid.close()
    
    # Export data file
    DC.writeUBC_DCobs(inv_dir + dsep + obsfile2d,Tx2d,Rx2d,data,unct,'2D') 
    
    # Write input file
    fid = open(inv_dir + dsep + inp_file,'w')
    fid.write('OBS LOC_X %s \n'% obsfile2d)
    fid.write('MESH FILE %s \n'% mshfile2d)
    fid.write('CHIFACT 1 %f\n'% chifact)
    fid.write('TOPO DEFAULT  %s \n')
    fid.write('INIT_MOD DEFAULT\n')
    fid.write('REF_MOD VALUE %e\n'% ref_mod)
    fid.write('ALPHA DEFAULT\n')
    fid.write('WEIGHT DEFAULT\n')
    fid.write('STORE_ALL_MODELS FALSE\n')
    fid.write('INVMODE SVD\n')
    fid.write('USE_MREF TRUE\n')
    fid.close()
    
    os.chdir(inv_dir)
    os.system('dcinv2d ' + inp_file)
    
    #%%
    #Load model
    minv = DC.readUBC_DC2DModel(inv_dir + dsep + 'dcinv2d.con')
    #plt.figure()
    axs = plt.subplot(2,1,2)
    
    plt.xlim([0,nc*dx])
    plt.ylim([mesh2d.vectorNy[-1]-dl_len,mesh2d.vectorNy[-1]])
    plt.gca().set_aspect('equal', adjustable='box')
    
    minv = np.reshape(minv,(mesh2d.nCy,mesh2d.nCx))
    plt.pcolormesh(mesh2d.vectorNx,mesh2d.vectorNy,np.log10(m2D),alpha=0.5, cmap='gray')
    plt.pcolormesh(mesh2d.vectorNx,mesh2d.vectorNy,np.log10(minv),alpha=0.5, clim=(np.min(np.log10(m2D)),np.max(np.log10(m2D))))
    cbar = plt.colorbar(format = '%.2f',fraction=0.02)
    cmin,cmax = cbar.get_clim()
    ticks = np.linspace(cmin,cmax,3)
    cbar.set_ticks(ticks)

#%% Othrwise it is a gradient array, plot surface of apparent resisitivty
elif re.match(stype,'gradient'):
    
    rC1P1 = np.sqrt( np.sum( (npm.repmat(Tx[0][0:2],Rx[0].shape[0], 1) - Rx[0][:,0:2])**2, axis=1 ))
    rC2P1 = np.sqrt( np.sum( (npm.repmat(Tx[0][3:5],Rx[0].shape[0], 1) - Rx[0][:,0:2])**2, axis=1 ))
    rC1P2 = np.sqrt( np.sum( (npm.repmat(Tx[0][0:2],Rx[0].shape[0], 1) - Rx[1][:,0:2])**2, axis=1 ))
    rC2P2 = np.sqrt( np.sum( (npm.repmat(Tx[0][3:5],Rx[0].shape[0], 1) - Rx[1][:,0:2])**2, axis=1 )) 
    
    rC1C2 = np.sqrt( np.sum( (npm.repmat(Tx[0][0:2]-Tx[0][3:5],Rx[0].shape[0], 1) )**2, axis=1 ))
    rP1P2 = np.sqrt( np.sum( (Rx[0][:,0:2] - Rx[1][:,0:2])**2, axis=1 ))
    
    rho = np.abs(data) *np.pi *2./ ( 1/rC1P1 - 1/rC2P1 - 1/rC1P2 + 1/rC2P2 )#*((rC1P1)**2 / rP1P2)# 

    Pmid = (Rx[0][:,0:2] + Rx[1][:,0:2])/2  
 
    # Grid points
    grid_x, grid_z = np.mgrid[np.min([Rx[0][:,0],Rx[1][:,0]]):np.max([Rx[0][:,0],Rx[1][:,0]]):a/10, np.min([Rx[0][:,0],Rx[1][:,1]]):np.max([Rx[0][:,0],Rx[1][:,1]]):a/10]
    grid_rho = griddata(np.c_[Pmid[:,0],Pmid[:,1]], (abs(1/rho.T)), (grid_x, grid_z), method='linear')
    
    
    #plt.subplot(2,1,2)
    plt.figure()
    plt.imshow(grid_rho.T, extent = (np.min(grid_x),np.max(grid_x),np.min(grid_z),np.max(grid_z))  ,origin='lower')
    
    var = 'Gradient Array - a-spacing: ' + str(a) + ' m'
    plt.title(var)
    plt.colorbar()
    plt.contour(grid_x,grid_z,grid_rho, colors='k')
    
    DCUtils.writeUBC_DCobs(home_dir+'\FWR_data3D.dat',survey,'3D','SURFACE')