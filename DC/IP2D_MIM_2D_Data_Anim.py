import os
from SimPEG import *
import SimPEG.DCIP as DC
import pylab as plt
from matplotlib import animation
from JSAnimation import HTMLWriter
from scipy.interpolate import NearestNDInterpolator
#from readUBC_DC2DMesh import readUBC_DC2DMesh
#from readUBC_DC2DModel import readUBC_DC2DModel
#from readUBC_DC2DLoc import readUBC_DC2DLoc
#from convertObs_DC3D_to_2D import convertObs_DC3D_to_2D
#from readUBC_DC3Dobs import readUBC_DC3Dobs

#%%
home_dir = 'C:\\Users\\dominiquef.MIRAGEOSCIENCE\\ownCloud\\Research\\MtIsa\\Data'
#msh_file = 'Mesh_2D.msh'
#mod_file = 'Model_2D.con'
obs_file ='ip3d_all.ip'
#obs_file = 'IP_Obs_QC_v6.dat'
dsep = '\\'


# Forward solver

# Number of padding cells to remove from plotting
padc = 15

# Plotting parameters
xmin, xmax = 10500, 13000
zmin, zmax = -600, 600
vmin, vmax = 0, 75

z = np.linspace(zmin, zmax, 4)

#%% load obs file 3D
dobs = DC.readUBC_DC3Dobs(home_dir + dsep + obs_file, rtype='IP')

DCsurvey = dobs['DCsurvey']
# Assign line ID to the survey
lineID = DC.xy_2_lineID(DCsurvey)
uniqueID = np.unique(lineID)    

# Convert 3D locations to 2D survey
dobs2D = DC.convertObs_DC3D_to_2D(DCsurvey, lineID,'Xloc')

srcMat = DC.getSrc_locs(DCsurvey)
#DCdata[src0, src0.rxList[0]]

# Find 2D data correspondance
dataID = np.zeros(dobs2D.nD)
count = 0
for ii in range(dobs2D.nSrc):
    nD = dobs2D.srcList[ii].rxList[0].nD
    dataID[count:count+nD] = ii
    count += nD
#==============================================================================
fig = plt.figure(figsize=(6,5))
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2)

def animate(ii):
    
    # Grab current line and 
    indx = np.where(lineID==ii)[0]
        
    srcLeft = []
    obs_l = []
    std_l = []
    
    srcRight = []
    obs_r = []
    std_r = []
    
    # Split the obs file into left and right
    for jj in range(len(indx)):
        
        # Grab corresponding data
        obs = dobs2D.dobs[dataID==indx[jj]]
        std = dobs2D.std[dataID==indx[jj]]
    
        Tx = dobs2D.srcList[indx[jj]].loc
        Rx = dobs2D.srcList[indx[jj]].rxList[0].locs
                
        # Create mid-point location
        Cmid = (Tx[0][0] + Tx[1][0])/2
        Pmid = (Rx[0][:,0] + Rx[1][:,0])/2
                
        ileft = Pmid < Cmid
        iright = Pmid >= Cmid
    
        if np.any(ileft):
            rx = DC.RxDipole(Rx[0][ileft,:],Rx[1][ileft,:])
            srcLeft.append( DC.SrcDipole( [rx], Tx[0],Tx[1] ) )
            
            obs_l = np.hstack([obs_l,obs[ileft]])
            std_l = np.hstack([std_l,std[ileft]])
                                         
        if np.any(iright):
            rx = DC.RxDipole(Rx[0][iright,:],Rx[1][iright,:])
            srcRight.append( DC.SrcDipole( [rx], Tx[0],Tx[1] ) )  
            
            obs_r = np.hstack([obs_r,obs[iright]])
            std_r = np.hstack([std_r,std[iright]])
    
    DC2D_l = DC.SurveyDC(srcLeft)  
    DC2D_l.dobs = np.asarray(obs_l)
    DC2D_l.std = np.asarray(std_l)
    
    DC2D_r = DC.SurveyDC(srcRight)  
    DC2D_r.dobs = np.asarray(obs_r)
    DC2D_r.std = np.asarray(std_r)
                                      
    removeFrame()
    #DC.plot_pseudoSection(dobs2D,lineID, np.r_[0,1],'pdp')
    
    id_lbe = int(DCsurvey.srcList[indx[jj]].loc[0][1])
    
    global ax1, ax2, fig
    ax1 = plt.subplot(2,1,1)
    ph = DC.plot_pseudoSection(DC2D_l,ax1,stype = 'pdp', dtype = 'volt', colorbar=False)
    ax1.set_title('Observed DP-P', fontsize=10)  
    ax1.set_xticklabels([])
    plt.xlim([xmin,xmax])
    plt.ylim([zmin,zmax])
    plt.gca().set_aspect('equal', adjustable='box')
    z = np.linspace(np.min(ph[2]),np.max(ph[2]), 5)
    z_label = np.linspace(20,1, 5)
    ax1.set_yticks(map(int, z))
    ax1.set_yticklabels(map(str, map(int, z_label)),size=8)
    ax1.set_ylabel('n-spacing',fontsize=8)
    
    # Add colorbar
    pos =  ax1.get_position()
    cbarax = fig.add_axes([pos.x0 + 0.72 , pos.y0 + 0.05,  pos.width*0.05, pos.height*0.5])  ## the parameters are the specified position you set
    cb = fig.colorbar(ph[0],cax=cbarax, orientation="vertical", ticks=np.linspace(ph[0].get_clim()[0],ph[0].get_clim()[1], 3), format="%4.1f")
    cb.set_label("App. Charg.",size=8)
    
    ax2 = plt.subplot(2,1,2)    
    ph = DC.plot_pseudoSection(DC2D_r,ax2,stype = 'pdp', dtype = 'volt', colorbar=False, clim = (ph[0].get_clim()[0],ph[0].get_clim()[1]))
    pos =  ax2.get_position()    
    ax2.set_position([pos.x0 , pos.y0,  pos.width, pos.height])
    plt.xlim([xmin,xmax])
    plt.ylim([zmin,zmax])
    plt.gca().set_aspect('equal', adjustable='box')
    ax2.set_title('Observed P-DP', fontsize=10)
    ax2.set_xlabel('Easting (m)', fontsize=8)
    z = np.linspace(np.min(ph[2]),np.max(ph[2]), 5)
    z_label = np.linspace(20,1, 5)
    ax2.set_yticks(map(int, z))
    ax2.set_yticklabels(map(str, map(int, z_label)),size=8)
    ax2.set_ylabel('n-spacing',fontsize=8)
    # Add colorbar
    pos =  ax2.get_position()
    cbarax = fig.add_axes([pos.x0 + 0.72 , pos.y0 + 0.05,  pos.width*0.05, pos.height*0.5])  ## the parameters are the specified position you set
    cb = fig.colorbar(ph[0],cax=cbarax, orientation="vertical", ticks=np.linspace(ph[0].get_clim()[0],ph[0].get_clim()[1], 3), format="%4.1f")
    cb.set_label("App. Charg.",size=8)
            
    

    
    bbox_props = dict(boxstyle="circle,pad=0.3",fc="r", ec="k", lw=1)
    ax2.text(0.00, 1, 'A', transform=ax2.transAxes, ha="left", va="center",
                size=6,
                bbox=bbox_props)
                
    bbox_props = dict(boxstyle="circle,pad=0.3",fc="y", ec="k", lw=1)
    ax2.text(0.1, 1, 'M', transform=ax2.transAxes, ha="left", va="center",
                size=6,
                bbox=bbox_props)
                
    bbox_props = dict(boxstyle="circle,pad=0.3",fc="g", ec="k", lw=1)
    ax2.text(0.2, 1, 'N', transform=ax2.transAxes, ha="left", va="center",
                size=6,
                bbox=bbox_props)
 
    bbox_props = dict(boxstyle="circle,pad=0.3",fc="g", ec="k", lw=1)
    ax1.text(0.00, 1, 'N', transform=ax1.transAxes, ha="left", va="center",
                size=6,
                bbox=bbox_props)
                
    bbox_props = dict(boxstyle="circle,pad=0.3",fc="y", ec="k", lw=1)
    ax1.text(0.1, 1, 'M', transform=ax1.transAxes, ha="left", va="center",
                size=6,
                bbox=bbox_props)
                
    bbox_props = dict(boxstyle="circle,pad=0.3",fc="r", ec="k", lw=1)
    ax1.text(0.2, 1, 'A', transform=ax1.transAxes, ha="left", va="center",
                size=6,
                bbox=bbox_props)
                

    

    #ax2.labelsize(fontsize=10)

#==============================================================================
#     ax2.annotate(str(id_lbe), xy=(0.0, float(ii)/len(uniqueID)), xycoords='figure fraction', 
#                 xytext=(0.01, float(ii)/len(uniqueID)), textcoords='figure fraction',
#                 arrowprops=dict(facecolor='black'),rotation=90)
#==============================================================================
            
    bbox_props = dict(boxstyle="rarrow,pad=0.3",fc="w", ec="k", lw=2)
    ax2.text(0.01, (float(ii)+1.)/(len(uniqueID)+2), 'N: ' + str(id_lbe), transform=fig.transFigure, ha="left", va="center",
                size=8,
                bbox=bbox_props)
    
    mrk_props = dict(boxstyle="square,pad=0.3",fc="w", ec="k", lw=2)
    ax2.text(0.01, 0.9, 'Line ID#', transform=fig.transFigure, ha="left", va="center",
                size=8,
                bbox=mrk_props)
                
    mrk_props = dict(boxstyle="square,pad=0.3",fc="b", ec="k", lw=2)
    
    for jj in range(len(uniqueID)):
        ax2.text(0.125, (float(jj)+1.)/(len(uniqueID)+2), ".", transform=fig.transFigure, ha="right", va="center",
                size=8,
                bbox=mrk_props)
                
    mrk_props = dict(boxstyle="square,pad=0.3",fc="r", ec="k", lw=2)
                
    ax2.text(0.125, (float(ii)+1.)/(len(uniqueID)+2), ".", transform=fig.transFigure, ha="right", va="center",
                size=8,
                bbox=mrk_props)  
                

def removeFrame():
    global ax1, ax2, fig
    fig.delaxes(ax1)
    fig.delaxes(ax2)
    #fig.delaxes(ax3)
    plt.draw() 

anim = animation.FuncAnimation(fig, animate, repeat = False,
                               frames=len(uniqueID), interval=1000)    
anim.save(home_dir + '\\animation.html', writer=HTMLWriter(embed_frames=True,fps=1))
