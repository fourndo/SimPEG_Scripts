import os
from SimPEG import *
import SimPEG.DCIP as DC
import pylab as plt
from matplotlib import animation
from JSAnimation import HTMLWriter
from pylab import get_current_fig_manager
from scipy.interpolate import NearestNDInterpolator
#%%
home_dir = 'C:\\Users\\dominiquef.MIRAGEOSCIENCE\\ownCloud\\Research\\MtIsa\\Data'
obs_file = 'ip3d_all_QC.ip'
topo_file  = 'MIM_SRTM_Local.topo'
dsep = '\\'

outfile = 'ip3d_all_QC.ip'

topo = np.genfromtxt(home_dir + dsep + topo_file,skip_header=1)
Ftopo = NearestNDInterpolator(topo[:,:2], topo[:,2])

dtype = 'volt'
plt.close('all')
#%% load obs file 3D
survey3D = DC.readUBC_DC3Dobs(home_dir + dsep + obs_file, rtype = 'IP')
DCsurvey = survey3D['DCsurvey']

# Data convertion to Chargeability
#DCsurvey.dobs = DCsurvey.dobs*100.
#DCsurvey.std = DCsurvey.std*100.

# Assign Z-value from topo
for ii in range(DCsurvey.nSrc):
    DCsurvey.srcList[ii].loc[0][2] = Ftopo(DCsurvey.srcList[ii].loc[0][0:2])
    DCsurvey.srcList[ii].loc[1][2] = Ftopo(DCsurvey.srcList[ii].loc[1][0:2])
    
    rx_x = DCsurvey.srcList[ii].rxList[0].locs[0][:,0]
    rx_y = DCsurvey.srcList[ii].rxList[0].locs[0][:,1]
    DCsurvey.srcList[ii].rxList[0].locs[0][:,2] = Ftopo(rx_x,rx_y)
    
    rx_x = DCsurvey.srcList[ii].rxList[0].locs[1][:,0]
    rx_y = DCsurvey.srcList[ii].rxList[0].locs[1][:,1]
    DCsurvey.srcList[ii].rxList[0].locs[1][:,2] = Ftopo(rx_x,rx_y)

# Assign line ID to the survey
lineID = DC.xy_2_lineID(DCsurvey)
uniqueID = np.unique(lineID)

# Convert 3D locations to 2D survey
dobs2D = DC.convertObs_DC3D_to_2D(DCsurvey, lineID,'Xloc')

srcMat = DC.getSrc_locs(DCsurvey)

# Find 2D data correspondance
dataID = np.zeros(dobs2D.nD)
count = 0
for ii in range(dobs2D.nSrc):
    nD = dobs2D.srcList[ii].rxList[0].nD
    dataID[count:count+nD] = ii
    count += nD


#%%
fig = plt.figure(figsize=(8,6))
ax1 = plt.subplot(1,1,1)
def removeFrame():
    global ax1, fig
    fig.delaxes(ax1)
    
#
#ax1 = plt.subplot(1,1,1)

#%%
uncert = []
uID = np.unique(lineID)
keeper = np.ones(len(dobs2D.dobs))

#len(uID)
for ii in range(len(uID)):
    
    # Grab all sources on current line and extract data
    indx = np.where(lineID==ii)[0]

    srcLeft, srcRight = [], []
    obs_l, obs_r = [], [] # Obs for the left-right data
    std_l, std_r = [], [] # Std-dev for the left-right data
    ind_l, ind_r = [], [] # Keep track of obs index from the global dataset

    # Split the obs file into left and right
    for jj in range(len(indx)):
        
        # Get numerical index of current data
        ind_data = np.where(dataID==indx[jj])[0]

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
            ind_l = np.hstack([ind_l,ind_data[ileft]])
                           
        if np.any(iright):
            rx = DC.RxDipole(Rx[0][iright,:],Rx[1][iright,:])
            srcRight.append( DC.SrcDipole( [rx], Tx[0],Tx[1] ) )  
            
            obs_r = np.hstack([obs_r,obs[iright]])
            std_r = np.hstack([std_r,std[iright]])
            ind_r = np.hstack([ind_r,ind_data[iright]])
    
 
    DC2D_l = DC.SurveyDC(srcLeft)  
    DC2D_l.dobs = np.asarray(obs_l)
    DC2D_l.std = np.asarray(std_l)
    
    DC2D_r = DC.SurveyDC(srcRight)  
    DC2D_r.dobs = np.asarray(obs_r)
    DC2D_r.std = np.asarray(std_r)
                                      
    #DC.plot_pseudoSection(dobs2D,lineID, np.r_[0,1],'pdp')
    for kk in range(2):
        
        removeFrame()
        # Get the survey and data index for either the right or left config
        if kk == 0:  
            survey = DC2D_l
            indx = ind_l
            
        if kk == 1:
            survey = DC2D_r
            indx = ind_r
            
#        id_lbe = int(survey.srcList[indx[jj]].loc[0][1])
            
        midx = []
        midz = []
        for jj in range(survey.nSrc):
            Tx = survey.srcList[jj].loc
            Rx = survey.srcList[jj].rxList[0].locs
    
            # Create mid-point location
            Cmid = (Tx[0][0] + Tx[1][0])/2
            Pmid = (Rx[0][:,0] + Rx[1][:,0])/2
        
            midx = np.hstack([midx, ( Cmid + Pmid )/2 ])
            midz = np.hstack([midz, -np.abs(Cmid-Pmid)/2 + (Tx[0][2] + Tx[1][2])/2 ])
        
        global ax1, fig
        ax1 = plt.subplot(1,1,1)
        
        
        ph = DC.plot_pseudoSection(survey,ax1,stype = 'pdp', dtype = dtype, colorbar=True)
        
        # Call a ginput to delete points on pseudo-section
        #uncert = np.hstack([uncert,src_uncert])
        cfm1=get_current_fig_manager().window  
        cfm1.activateWindow()
        plt.sca(ax1)
        gin = plt.ginput(100, timeout = 0, show_clicks=True,mouse_add=1,mouse_stop=2)
        
        # Find closest midx and midz from the list of gin and assign 0 to
        #the global data indexing
        for jj in range(len(gin)):
            
            # Find the closest point on the pseudo section
            rmin = np.argmin( (gin[jj][0] - midx)**2. + (gin[jj][1] - midz)**2. )
            keeper[indx[rmin]] = 0
        

#%% Reconstruct the 3D survey minus the discarted data
srcList = []
for ii in range(DCsurvey.nSrc):
        
    Tx = DCsurvey.srcList[ii].loc
    
    indx = np.where(keeper[dataID==ii]==1)[0]

    rx = []
    # Make sure that transmitter is not empty
    if len(indx)!=0:
            
        # Construct a list of receivers
        rx = DC.RxDipole(DCsurvey.srcList[ii].rxList[0].locs[0][indx,:],
                         DCsurvey.srcList[ii].rxList[0].locs[1][indx,:])
        srcList.append( DC.SrcDipole( [rx], Tx[0],Tx[1] ) ) 
    
DCsurvey_out = DC.SurveyDC(srcList)  
DCsurvey_out.dobs = DCsurvey.dobs[np.where(keeper==1)[0]]
DCsurvey_out.std = DCsurvey.std[np.where(keeper==1)[0]]
# Replace the uncertainties on the orginal survey
#DCsurvey.uncert = uncert

# Write new obsfile out
DC.writeUBC_DCobs(home_dir+dsep+outfile,DCsurvey_out,'3D', surveyType='GENERAL', iptype=1)

