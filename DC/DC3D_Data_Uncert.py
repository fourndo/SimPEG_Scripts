import os
from SimPEG import *
import SimPEG.DCIP as DC
import pylab as plt
from matplotlib import animation
from JSAnimation import HTMLWriter


#%%
home_dir = 'C:\\Users\\dominiquef.MIRAGEOSCIENCE\\ownCloud\\Research\\MtIsa\\Data'
obs_file = 'IP_Obs_QC_v5.dat'
dsep = '\\'

#%% load obs file 3D
survey3D = DC.readUBC_DC3Dobs(home_dir + dsep + obs_file, dtype = 'IP')

DCsurvey = survey3D['DCsurvey']

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


uncert = []
uID = np.unique(lineID)

for ii in range(len(uID)):
    
    # Grab all sources on current line and extract data
    indx = np.where(lineID==ii)[0]

    # Split the obs file into left and right
    for jj in range(len(indx)):
    
        # Grab corresponding data
        src_obs = dobs2D.dobs[dataID==indx[jj]]
        src_uncert = src_obs*0.         
        
        Tx = dobs2D.srcList[indx[jj]].loc
        Rx = dobs2D.srcList[indx[jj]].rxList[0].locs
    
        # Create mid-point location
        Cmid = (Tx[0][0] + Tx[1][0])/2
        Pmid = (Rx[0][:,0] + Rx[1][:,0])/2
    
        ileft = Pmid < Cmid
        iright = Pmid >= Cmid
        
#        src_uncert[iright] = np.abs(np.asarray(src_obs[iright]))*0.02 + 2e-5
#        src_uncert[ileft] = np.abs(np.asarray(src_obs[ileft]))*0.06 + 4e-5
        src_uncert[iright] = np.abs(np.asarray(src_obs[iright]))*0.1 + 2e+0
        src_uncert[ileft] = np.abs(np.asarray(src_obs[ileft]))*0.1 + 2e+0
        uncert = np.hstack([uncert,src_uncert])

# Replace the uncertainties on the orginal survey
DCsurvey.std = uncert

# Write new obsfile out
DC.writeUBC_DCobs(home_dir+dsep+'IP3D_Obs_Data_Uncert.dat',DCsurvey,stype='GENERAL', iptype=1)
