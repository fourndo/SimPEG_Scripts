import os
from SimPEG import *
import SimPEG.DCIP as DC
import pylab as plt
from matplotlib import animation
from JSAnimation import HTMLWriter
import scipy.interpolate as interpolation
#from readUBC_DC2DMesh import readUBC_DC2DMesh
#from readUBC_DC2DModel import readUBC_DC2DModel
#from readUBC_DC2DLoc import readUBC_DC2DLoc
#from convertObs_DC3D_to_2D import convertObs_DC3D_to_2D
#from readUBC_DC3Dobs import readUBC_DC3Dobs

#%%
home_dir = 'C:\\Users\\dominiquef.MIRAGEOSCIENCE\\ownCloud\\Research\\MtIsa\\Data'
#msh_file = 'Mesh_2D.msh'
#mod_file = 'Model_2D.con'
obs_file = 'data_Z.txt'
topofile = 'MIM_SRTM_Local.topo'
dsep = '\\'

# Forward solver

# Number of padding cells to remove from plotting
padc = 15

# Load UBC-topo file
topo = np.genfromtxt(home_dir + dsep + topofile, skip_header = 1)

# load obs file 3D
dobs = DC.readUBC_DC3Dobs(home_dir + dsep + obs_file)

DCsurvey = dobs['DCsurvey']
# Assign line ID to the survey
lineID = DC.xy_2_lineID(DCsurvey)
uniqueID = np.unique(lineID)

# Convert 3D locations to 2D survey
dobs2D = DC.convertObs_DC3D_to_2D(DCsurvey, lineID, 'Xloc')

srcMat = DC.getSrc_locs(dobs2D)
#DCdata[src0, src0.rxList[0]]

# Find 2D data correspondance
dataID = np.zeros(dobs2D.nD)
count = 0
for ii in range(dobs2D.nSrc):
    nD = dobs2D.srcList[ii].rxList[0].nD
    dataID[count:count+nD] = ii
    count += nD
#==============================================================================

#%% Create dcin2d inversion files and run
dx = 20.
    
dl_len = np.max(srcMat[lineID==0,0,0]) - np.min(srcMat[lineID==0,0,0])
nc = np.ceil(dl_len/dx)+10

padx = dx*np.power(1.4,range(1,padc))

# Creating padding cells
hx = np.r_[padx[::-1], np.ones(nc)*dx , padx]
hz = np.r_[padx[::-1], np.ones(int(nc/3))*dx]

# Create mesh with 0 coordinate centerer on the ginput points in cell center
mesh2d = Mesh.TensorMesh([hx, hz], x0=(-np.sum(padx) + np.min(srcMat[0][:,0]) - dx/2,np.max(srcMat[0][0,2])-np.sum(hz)))
    
inv_dir = home_dir + '\Inv2D' 
if not os.path.exists(inv_dir):
    os.makedirs(inv_dir)
    
mshfile2d = 'Mesh_2D.msh'
modfile2d = 'MtIsa_2D.con'
obsfile2d = 'FWR_3D_2_2D.dat'
inp_file = 'dcinv2d.inp'

ini_mod = 1e-2
ref_mod = 1e-2

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

#%% Run all lines and store

mout = []
dout = []
for ii in range(1):
    
    # Grab current line and 
    indx = np.where(lineID==ii)[0]
        
    srcList = []
    d = []
    std =[]
    for ss in range(len(indx)):
        Rx = dobs2D.srcList[indx[ss]].rxList[0]
        Tx = dobs2D.srcList[indx[ss]].loc
        d.extend(mkvc(dobs2D.dobs[dataID==indx[ss]]))
        std.extend(dobs2D.std[dataID==indx[ss]])        
        srcList.append(DC.SrcDipole([Rx],Tx[0],Tx[1]))
    
    data =np.array([np.array(xi) for xi in d])
    unct =np.array([np.array(xi) for xi in std])
    DC2D = DC.SurveyDC(srcList)  
    DC2D.dobs = data
    DC2D.std = unct

    #DC.plot_pseudoSection(dobs2D,lineID, np.r_[0,1],'pdp')
    
    id_lbe = int(DCsurvey.srcList[indx[ss]].loc[0][1])
    global ax1, fig


    ax1 = plt.subplot(1,1,1)    
    for ss in range(DC2D.nSrc):
        Tx = DC2D.srcList[ss].loc[0]
        plt.scatter(Tx[0],Tx[2],s=10)
    
    # Create topo file if not empty
    if topo is not None:
        F = interpolation.NearestNDInterpolator(np.c_[topo[:,0],topo[:,1]],topo[:,2])
        topo2D = F(mesh2d.vectorCCx,id_lbe)
        
        # Export topography file
        with file(inv_dir + dsep + 'topofile.dat','w') as fid:
            fid.write('%i %e\n'% (topo2D.shape[0],np.max(topo2D)))
            np.savetxt(fid, np.c_[mesh2d.vectorCCx,topo2D], fmt='%e',delimiter=' ',newline='\n')
        
    # Export data file
    DC.writeUBC_DCobs(inv_dir + dsep + obsfile2d,DC2D,'2D','SIMPLE') 
    
    # Write input file
    fid = open(inv_dir + dsep + inp_file,'w')
    fid.write('OBS LOC_X %s \n'% obsfile2d)
    fid.write('MESH FILE %s \n'% mshfile2d)
    fid.write('CHIFACT 1 \n')
    fid.write('TOPO FILE topofile.dat \n')
    fid.write('INIT_MOD VALUE %e\n'% ini_mod)
    fid.write('REF_MOD VALUE %e\n'% ref_mod)
    fid.write('ALPHA DEFAULT\n')
    fid.write('WEIGHT DEFAULT\n')
    fid.write('STORE_ALL_MODELS FALSE\n')
    fid.write('INVMODE SVD\n')
    fid.write('USE_MREF TRUE\n')
    fid.close()
    
    print ii
    os.chdir(inv_dir)
    os.system('dcinv2d ' + inp_file)
    
    #%%    
    
    #Load model
    mout.append( DC.readUBC_DC2DModel(inv_dir + dsep + 'dcinv2d.con'))
    dpre = DC.readUBC_DC2Dpre(inv_dir + dsep + 'dcinv2d.pre') 

    dout.append( dpre['DCsurvey'] )
    
    #DCpre.dobs = (DC2D.dobs - DCpre.dobs) / DC2D.std
    

#%% Add the extra
# Cycle through the lines and invert len(uniqueID)
fig = plt.figure(figsize=(8,5))
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)


def animate(ii):
    
    
    removeFrame2()
    
    global ax1, ax2, fig
    
    minv = mout[ii]
    dpre = dout[ii]
    
    #airind = minv==1e-8
    #minv[airind] = np.nan

    ax1 = plt.subplot(1,2,1)
    ax1.set_title('2D Conductivity (S/m)', fontsize=10) 
    plt.xlim([mesh2d.vectorNx[padc],mesh2d.vectorNx[-padc]])
    plt.ylim([mesh2d.vectorNy[-1]-dl_len/3,mesh2d.vectorNy[-1]+60])
    plt.gca().set_aspect('equal', adjustable='box')
    
    minv = np.reshape(minv,(mesh2d.nCy,mesh2d.nCx))
    #plt.pcolormesh(mesh2d.vectorNx,mesh2d.vectorNy,np.log10(m2D),alpha=0.5, cmap='gray')
    plt.pcolormesh(mesh2d.vectorNx,mesh2d.vectorNy,np.log10(minv), vmin=-4,vmax=2)
    plt.gca().tick_params(axis='both', which='major', labelsize=8)
    ax1.yaxis.tick_right() 
    
    
    cbar = plt.colorbar(format = '%.2f',fraction=0.03,orientation="horizontal")
    cmin,cmax = cbar.get_clim()
    ticks = np.linspace(cmin,cmax,3)
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=10)

    ax2 = plt.subplot(1,2,2)
    ax2 = DC.plot_pseudoSection(dpre,ax2,'pdp')#axs.pcolormesh(mesh_sub.vectorCCx,mesh_sub.vectorCCy,Q_sub, alpha=0.75,vmin=-1e-2, vmax=1e-2)
    ax2.set_title('App Cond (S/m)', fontsize=10) 

    plt.draw()  
    bbox_props = dict(boxstyle="rarrow,pad=0.3",fc="w", ec="k", lw=2)
    ax1.text(0.01, (float(ii)+1.)/(len(uniqueID)+2), 'N: ' + str(id_lbe), transform=fig.transFigure, ha="left", va="center",
                size=8,
                bbox=bbox_props)
    
    mrk_props = dict(boxstyle="square,pad=0.3",fc="b", ec="k", lw=2)
    
    for jj in range(len(uniqueID)):
        ax1.text(0.1, (float(jj)+1.)/(len(uniqueID)+2), ".", transform=fig.transFigure, ha="right", va="center",
                size=8,
                bbox=mrk_props)
                
    mrk_props = dict(boxstyle="square,pad=0.3",fc="r", ec="k", lw=2)
                
    ax1.text(0.1, (float(ii)+1.)/(len(uniqueID)+2), ".", transform=fig.transFigure, ha="right", va="center",
                size=8,
                bbox=mrk_props)  

def removeFrame2():
    global ax1, ax2, fig
    fig.delaxes(ax1)
    fig.delaxes(ax2)
    plt.draw() 

anim = animation.FuncAnimation(fig, animate,
                               frames=len(mout), interval=1000, repeat = False) 
                               
anim.save(home_dir + '\\Invmodels.html', writer=HTMLWriter(embed_frames=True,fps=1))



