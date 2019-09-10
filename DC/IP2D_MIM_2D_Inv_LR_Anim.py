import os
from SimPEG import Mesh, np
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
DCobs_file = 'data_Z.txt'
IPobs_file = 'ip3d_all_QC.ip'
topo_file  = 'MIM_SRTM_Local.topo'
dsep = '\\'

topo = np.genfromtxt(home_dir + dsep + topo_file,skip_header=1)
Ftopo = NearestNDInterpolator(topo[:,:2], topo[:,2])


# Forward solver

# Number of padding cells to remove from plotting
padc = 15

# Plotting parameters
xmin, xmax = 10500, 13000
zmin, zmax = -600, 600
vmin, vmax = 0, 50

z = np.linspace(zmin, zmax, 4)
x = np.asarray([11000,11750,12500])
#%% load obs file 3D
dobs = DC.readUBC_DC3Dobs(home_dir + dsep + DCobs_file)
DCsurvey = dobs['DCsurvey']

dobs = DC.readUBC_DC3Dobs(home_dir + dsep + IPobs_file, rtype = 'IP')
IPsurvey = dobs['DCsurvey']

# Assign Z-value from topo
for ii in range(IPsurvey.nSrc):
    IPsurvey.srcList[ii].loc[0][2] = Ftopo(IPsurvey.srcList[ii].loc[0][0:2])
    IPsurvey.srcList[ii].loc[1][2] = Ftopo(IPsurvey.srcList[ii].loc[1][0:2])
    
    rx_x = IPsurvey.srcList[ii].rxList[0].locs[0][:,0]
    rx_y = IPsurvey.srcList[ii].rxList[0].locs[0][:,1]
    IPsurvey.srcList[ii].rxList[0].locs[0][:,2] = Ftopo(rx_x,rx_y)
    
    rx_x = IPsurvey.srcList[ii].rxList[0].locs[1][:,0]
    rx_y = IPsurvey.srcList[ii].rxList[0].locs[1][:,1]
    IPsurvey.srcList[ii].rxList[0].locs[1][:,2] = Ftopo(rx_x,rx_y)

# Assign line ID to the survey
lineID = DC.xy_2_lineID(DCsurvey)
uniqueID = np.unique(lineID)

IPlineID = DC.xy_2_lineID(IPsurvey)

# Convert 3D locations to 2D survey
DCdobs2D = DC.convertObs_DC3D_to_2D(DCsurvey, lineID,'Xloc')


IPdobs2D = DC.convertObs_DC3D_to_2D(IPsurvey, IPlineID,'Xloc')

srcMat = DC.getSrc_locs(IPsurvey)
#DCdata[src0, src0.rxList[0]]

# Find 2D data correspondance
dataID = np.zeros(DCdobs2D.nD)
count = 0
for ii in range(DCdobs2D.nSrc):
    nD = DCdobs2D.srcList[ii].rxList[0].nD
    dataID[count:count+nD] = ii
    count += nD

# Find 2D data correspondance
IPdataID = np.zeros(IPdobs2D.nD)
count = 0
for ii in range(IPdobs2D.nSrc):
    nD = IPdobs2D.srcList[ii].rxList[0].nD
    IPdataID[count:count+nD] = ii
    count += nD
    
#
#%% Create dcin2d inversion files and run
dx = 20.

dl_len = np.max(srcMat[lineID==0,0,0]) - np.min(srcMat[lineID==0,0,0])
nc = np.ceil(dl_len/dx)+10

padx = dx*np.power(1.4,range(1,padc))

# Creating padding cells
hx = np.r_[padx[::-1], np.ones(nc)*dx , padx]
hz = np.r_[padx[::-1], np.ones(int(nc/5))*dx]

# Create mesh with 0 coordinate centerer on the ginput points in cell center
mesh2d = Mesh.TensorMesh([hx, hz], x0=(-np.sum(padx) + np.min(srcMat[0][:,0]),np.ceil(np.max(srcMat[0][0,2]))-np.sum(hz)))

inv_dir = home_dir + '\Inv2D'
if not os.path.exists(inv_dir):
    os.makedirs(inv_dir)

mshfile2d = 'Mesh_2D.msh'
modfile2d = 'MtIsa_2D.con'
obsfile2d = 'FWR_3D_2_2D.dat'
ipfile2d = 'IP_2D.dat'
inp_file = 'dcinv2d.inp'

ini_mod = 1e-2
ref_mod = 1e-2

#%% Export 2D mesh
fid = open(inv_dir + dsep + mshfile2d,'w')
fid.write('%i\n'% mesh2d.nCx)
fid.write('%f %f 1\n'% (mesh2d.vectorNx[0],mesh2d.vectorNx[1]))
np.savetxt(fid, np.c_[mesh2d.vectorNx[2:],np.ones(mesh2d.nCx-1)], fmt='\t %e %i',delimiter=' ',newline='\n')
fid.write('\n')
fid.write('%i\n'% mesh2d.nCy)
fid.write('%f %f 1\n'%( -mesh2d.vectorNy[-1], -mesh2d.vectorNy[-2]))
np.savetxt(fid, np.c_[-mesh2d.vectorNy[-3::-1],np.ones(mesh2d.nCy-1)], fmt='\t %f %i',delimiter=' ',newline='\n')
fid.close()

#==============================================================================
# # Export 2D model
# fid = open(inv_dir + dsep + modfile2d,'w')
# fid.write('%i %i\n'% (mesh2d.nCx,mesh2d.nCy))
# np.savetxt(fid, mkvc(m2D[::-1,:].T), fmt='%e',delimiter=' ',newline='\n')
# fid.close()
#==============================================================================

#%% Cycle through lines and invert len(uniqueID)
fig = plt.figure(figsize=(8,6))
ax1 = plt.subplot(3,2,1)
ax2 = plt.subplot(3,2,2)
#ax1, ax2 = DC.plot_pseudoSection(DCdobs2D,lineID, np.r_[0],'pdp')
ax3 = plt.subplot(3,2,3)
ax4 = plt.subplot(3,2,4)

ax5 = plt.subplot(3,2,5)
ax6 = plt.subplot(3,2,6)

def animate(ii):

    #for ii in range(1):
    removeFrame()
    # Grab current line and
    indx = np.where(lineID==ii)[0]
    
    srcLeft = []
    obs_l = []
    obs = []
    srcRight = []
    obs_r = []
    srcList=[]
    # Split the obs file into left and right
    # Split the obs file into left and right
    for jj in range(len(indx)):
    
        # Grab corresponding data
        obs = np.hstack([obs,DCdobs2D.dobs[dataID==indx[jj]]])
        #std = dobs2D.std[dataID==indx[jj]]
    
        srcList.append(DCdobs2D.srcList[indx[jj]])
        
        Tx = DCdobs2D.srcList[indx[jj]].loc
        Rx = DCdobs2D.srcList[indx[jj]].rxList[0].locs
    
        # Create mid-point location
        Cmid = (Tx[0][0] + Tx[1][0])/2
        Pmid = (Rx[0][:,0] + Rx[1][:,0])/2
    
        ileft = Pmid < Cmid
        iright = Pmid >= Cmid
        
        temp = np.zeros(len(ileft))
        temp[ileft] = 1
        obs_l = np.hstack([obs_l,temp])
            
        temp = np.zeros(len(iright))
        temp[iright] = 1
        obs_r = np.hstack([obs_r,temp])
        
        if np.any(ileft):
            rx = DC.RxDipole(Rx[0][ileft,:],Rx[1][ileft,:])
            srcLeft.append( DC.SrcDipole( [rx], Tx[0],Tx[1] ) )
    
            
            #std_l = np.hstack([std_l,std[ileft]])
    
        if np.any(iright):
            rx = DC.RxDipole(Rx[0][iright,:],Rx[1][iright,:])
            srcRight.append( DC.SrcDipole( [rx], Tx[0],Tx[1] ) )
    
            #obs_r = np.hstack([obs_r,iright])
            #std_r = np.hstack([std_r,std[iright]])
    
    
    DC2D_full = DC.SurveyDC(srcList)
    DC2D_full.dobs = np.asarray(obs)
    DC2D_full.std = DC2D_full.dobs*0.
    DC2D_full.std[obs_l==1] = np.abs(DC2D_full.dobs[obs_l==1])*0.02 + 2e-5
    DC2D_full.std[obs_r==1] = np.abs(DC2D_full.dobs[obs_r==1])*0.06 + 4e-5
    
#    DC2D_l = DC.SurveyDC(srcLeft)
#    DC2D_l.dobs = np.asarray(obs[obs_l==1])
#    DC2D_l.std = np.abs(np.asarray(DC2D_l.dobs))*0.05 + 2e-5
#    
#    DC2D_r = DC.SurveyDC(srcRight)
#    DC2D_r.dobs = np.asarray(obs[obs_r==1])
#    DC2D_r.std = np.abs(np.asarray(DC2D_r.dobs))*0.05 + 2e-5
    
    
    survey = DC2D_full
    
    # Export data file
    DC.writeUBC_DCobs(inv_dir + dsep + obsfile2d,survey,'2D','SIMPLE')
    
    # Write input file
    fid = open(inv_dir + dsep + inp_file,'w')
    fid.write('OBS LOC_X %s \n'% obsfile2d)
    fid.write('MESH FILE %s \n'% mshfile2d)
    fid.write('CHIFACT 6 \n')
    fid.write('TOPO DEFAULT \n')
    fid.write('INIT_MOD VALUE %e\n'% ini_mod)
    fid.write('REF_MOD VALUE %e\n'% ref_mod)
    fid.write('ALPHA VALUE %f %f %F\n'% (1./dx**4., 1, 1))
    fid.write('WEIGHT DEFAULT\n')
    fid.write('STORE_ALL_MODELS FALSE\n')
    fid.write('INVMODE CG\n')
    #fid.write('CG_PARAM 200 1e-4\n')
    fid.write('USE_MREF FALSE\n')
    #fid.write('BOUNDS VALUE 1e-4 1e+2\n')
    fid.close()
    
    os.chdir(inv_dir)
    os.system('dcinv2d ' + inp_file)
    
    #%% Load DC model and predicted data
    minv = DC.readUBC_DC2DModel(inv_dir + dsep + 'dcinv2d.con')
    minv = np.reshape(minv,(mesh2d.nCy,mesh2d.nCx))
    
    #%% Repeat for IP data
    indx = np.where(IPlineID==ii)[0]
    
    srcLeft = []
    obs_l = []
    std_l = []
    
    srcRight = []
    obs_r = []
    std_r = []
    
    # Split the obs file into left and right
    for jj in range(len(indx)):
    
        # Grab corresponding data
        obs = IPdobs2D.dobs[IPdataID==indx[jj]]
        std = IPdobs2D.std[IPdataID==indx[jj]]
    
        Tx = IPdobs2D.srcList[indx[jj]].loc
        Rx = IPdobs2D.srcList[indx[jj]].rxList[0].locs
    
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
    
    IP2D_l = DC.SurveyDC(srcLeft)
    IP2D_l.dobs = np.asarray(obs_l)
    IP2D_l.std = std_l#np.abs(np.asarray(obs_l))*0.1 + 2e+0
    
    IP2D_r = DC.SurveyDC(srcRight)
    IP2D_r.dobs = np.asarray(obs_r)
    IP2D_r.std = std_r#np.abs(np.asarray(obs_r))*0.1 + 2e+0
    
    id_lbe = int(IPsurvey.srcList[indx[jj]].loc[0][1])
    global ax1, ax2,ax3, ax4, ax5, ax6, fig
    
    ax2 = plt.subplot(3,2,2)
    ph = DC.plot_pseudoSection(IP2D_r,ax2,stype ='pdp', dtype = 'volt', colorbar=False)
    ax2.set_title('Observed P-DP', fontsize=10)
    plt.xlim([xmin,xmax])
    plt.ylim([zmin,zmax])
    plt.gca().set_aspect('equal', adjustable='box')
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    
    ax1 = plt.subplot(3,2,1)
    DC.plot_pseudoSection(IP2D_l,ax1,stype ='pdp', dtype = 'volt', clim = (ph[0].get_clim()[0],ph[0].get_clim()[1]), colorbar=False)
    ax1.set_title('Observed DP-P', fontsize=10)
    plt.xlim([xmin,xmax])
    plt.ylim([zmin,zmax])
    plt.gca().set_aspect('equal', adjustable='box')    
    z = np.linspace(np.min(ph[2]),np.max(ph[2]), 5)
    z_label = np.linspace(20,1, 5)
    ax1.set_yticks(map(int, z))
    ax1.set_yticklabels(map(str, map(int, z_label)),size=8)
    ax1.set_ylabel('n-spacing',fontsize=8)
            
    #%% Add labels
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
                
    # Run both left and right survey seperately
    for kk in range(2):
    
        if kk == 0:
    
            survey = IP2D_l
    
        if kk == 1:
    
            survey = IP2D_r
    
    
        # Export data file
        DC.writeUBC_DCobs(inv_dir + dsep + ipfile2d,survey,'2D','SIMPLE',iptype = 1)
    
        fid = open(inv_dir + dsep + inp_file,'w')
        fid.write('OBS LOC_X %s \n'% ipfile2d)
        fid.write('MESH FILE %s \n'% mshfile2d)
        

        fid.write('CHIFACT 4 \n')
            
        fid.write('COND FILE dcinv2d.con\n')
        fid.write('TOPO DEFAULT \n')
        
        fid.write('INIT_MOD VALUE %e\n'% ini_mod)
            
        fid.write('REF_MOD VALUE 0.0\n')
        fid.write('ALPHA VALUE %f %f %F\n'% (1./dx**4., 1, 1))
        fid.write('WEIGHT DEFAULT\n')
        fid.write('STORE_ALL_MODELS FALSE\n')
        fid.write('INVMODE CG\n')
        #fid.write('CG_PARAM 200 1e-4\n')
        fid.write('USE_MREF FALSE\n')
        #fid.write('BOUNDS VALUE 1e-4 1e+2\n')
        fid.close()
    
        os.chdir(inv_dir)
        os.system('ipinv2d ' + inp_file)
    
        #%% Load model and predicted data
        minv = DC.readUBC_DC2DModel(inv_dir + dsep + 'ipinv2d.chg')
        minv = np.reshape(minv,(mesh2d.nCy,mesh2d.nCx))
    
        dpre = DC.readUBC_DC2Dpre(inv_dir + dsep + 'ipinv2d.pre')
        DCpre = dpre['DCsurvey']
        DCtemp = survey
        DCtemp.dobs = DCpre.dobs
    
        if kk == 0:
    
            ax5 = plt.subplot(3,2,3)
            DC.plot_pseudoSection(DCtemp,ax5,stype ='pdp',dtype='volt', clim = (ph[0].get_clim()[0],ph[0].get_clim()[1]), colorbar=False)
            ax5.set_title('Predicted', fontsize=10)
            plt.xlim([xmin,xmax])
            plt.ylim([zmin,zmax])
            plt.gca().set_aspect('equal', adjustable='box')
            ax5.set_xticklabels([])
            z = np.linspace(np.min(ph[2]),np.max(ph[2]), 5)
            z_label = np.linspace(20,1, 5)
            ax5.set_yticks(map(int, z))
            ax5.set_yticklabels(map(str, map(int, z_label)),size=8)
            ax5.set_ylabel('n-spacing',fontsize=8)
            
        else:
    
            ax6 = plt.subplot(3,2,4)
            DC.plot_pseudoSection(DCtemp,ax6,stype ='pdp',dtype='volt', clim = (ph[0].get_clim()[0],ph[0].get_clim()[1]), colorbar=False)
            ax6.set_title('Predicted', fontsize=10)
            plt.xlim([xmin,xmax])
            plt.ylim([zmin,zmax])
            plt.gca().set_aspect('equal', adjustable='box')
            ax6.set_xticklabels([])
            ax6.set_yticklabels([])
            
            pos =  ax6.get_position()
            cbarax = fig.add_axes([pos.x0 + 0.325 , pos.y0 + 0.2,  pos.width*0.1, pos.height*0.5])  ## the parameters are the specified position you set
            cb = fig.colorbar(ph[0],cax=cbarax, orientation="vertical", ax = ax6, ticks=np.linspace(ph[0].get_clim()[0],ph[0].get_clim()[1], 4), format="%4.1f")
            cb.set_label("App. Charg.",size=8)
    
        if kk == 0:
            ax3 = plt.subplot(3,2,5)
            ax3.set_title('2-D Model (S/m)', fontsize=10)
            ax3.set_xticks(map(int, x))
            ax3.set_xticklabels(map(str, map(int, x)))
            ax3.set_xlabel('Easting (m)', fontsize=8)
            ax3.set_yticks(map(int, z))
            ax3.set_yticklabels(map(str, map(int, z)),rotation='vertical')
            ax3.set_ylabel('Depth (m)', fontsize=8)
            
        if kk == 1:
            ax4 = plt.subplot(3,2,6)
            ax4.set_title('2-D Model (S/m)', fontsize=10)
            ax4.set_xticks(map(int, x))
            ax4.set_xticklabels(map(str, map(int, x)))
            ax4.set_xlabel('Easting (m)', fontsize=8)
            ax4.set_yticklabels([])
    
        plt.xlim([xmin,xmax])
        plt.ylim([zmin/2,zmax])
        plt.gca().set_aspect('equal', adjustable='box')
    
        ph2 = plt.pcolormesh(mesh2d.vectorNx,mesh2d.vectorNy,(minv), vmin=vmin, vmax=vmax)
        plt.gca().tick_params(axis='both', which='major', labelsize=8)
    
        plt.draw()
    
        for ss in range(survey.nSrc):
            Tx = survey.srcList[ss].loc[0]
            plt.scatter(Tx[0],mesh2d.vectorNy[-1]+10,s=10)
    
    
        if kk == 1:
            
            pos =  ax4.get_position()
            cbarax = fig.add_axes([pos.x0 + 0.325 , pos.y0 + 0.015,  pos.width*0.1, pos.height*0.75])  ## the parameters are the specified position you set
            cb = fig.colorbar(ph2,cax=cbarax, orientation="vertical", ax = ax4, ticks=np.linspace(vmin,vmax, 4))
            cb.set_label("Chargeability",size=8)
    
    pos =  ax1.get_position()
    ax1.set_position([pos.x0 +0.03 , pos.y0,  pos.width, pos.height])
    
    pos =  ax5.get_position()
    ax5.set_position([pos.x0 +0.03 , pos.y0,  pos.width, pos.height])
    
    pos =  ax3.get_position()
    ax3.set_position([pos.x0 +0.03 , pos.y0,  pos.width, pos.height])
    
    pos =  ax2.get_position()
    ax2.set_position([pos.x0 -0.03 , pos.y0,  pos.width, pos.height])
    
    pos =  ax6.get_position()
    ax6.set_position([pos.x0 -0.03 , pos.y0,  pos.width, pos.height])
    
    pos =  ax4.get_position()
    ax4.set_position([pos.x0 -0.03 , pos.y0,  pos.width, pos.height])
    #%% Add the extra
    
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
        ax2.text(0.1, (float(jj)+1.)/(len(uniqueID)+2), ".", transform=fig.transFigure, ha="right", va="center",
                size=8,
                bbox=mrk_props)
    
    mrk_props = dict(boxstyle="square,pad=0.3",fc="r", ec="k", lw=2)
    
    ax2.text(0.1, (float(ii)+1.)/(len(uniqueID)+2), ".", transform=fig.transFigure, ha="right", va="center",
                size=8,
                bbox=mrk_props)

def removeFrame():
    global ax1, ax2,ax3,ax4, ax5, ax6, fig
    fig.delaxes(ax1)
    fig.delaxes(ax2)
    fig.delaxes(ax3)
    fig.delaxes(ax4)
    fig.delaxes(ax5)
    fig.delaxes(ax6)
    plt.draw()

anim = animation.FuncAnimation(fig, animate,
                               frames=len(uniqueID) , interval=1000, repeat = False)
#/home/dominiquef/3796_AGIC_Research/DCIP3D/MtISa
anim.save(home_dir + '\\Invmodels.html', writer=HTMLWriter(embed_frames=True,fps=1))



