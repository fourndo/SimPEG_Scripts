# -*- coding: utf-8 -*-
"""
Script to subsample a survey with distance

Created on Wed Sep  6 09:57:20 2017

@author: DominiqueFournier
"""

from SimPEG import Utils
from SimPEG.Utils import mkvc
import SimPEG.PF as PF
import numpy as np
import matplotlib.pyplot as plt
import csv as reader
from scipy.spatial import cKDTree

# # USER INPUTS # #
# workDir = "C:\\Users\\DominiqueFournier\\Dropbox\\Projects\\Kevitsa\\Kevitsa\\Modeling\\MAG\\Airborne"
workDir = r"C:\Users\DominiqueFournier\Downloads\data_dom"
dFile = "CUS_SL_2808ds50m.mag"
dType = 'MAG'
method = ('radius', 500)#('random', 0.2)  # #

dFileOut = 'CUS_SL_2808ds500m.mag'

# # SCRIPT STARTS HERE # #
if dType == 'MAG':
    survey, _ = Utils.io_utils.readUBCmagneticsObservations(workDir + '\\' + dFile)
    locXYZ = survey.srcField.rxList[0].locs
elif dType == 'GRAV':
    survey = Utils.io_utils.readUBCgravityObservations(workDir + '\\' + dFile)
    locXYZ = survey.srcField.rxList[0].locs
elif dType == 'XYZ':
    survey = np.loadtxt(workDir + "\\" + dFile, skiprows=1)
    locXYZ = survey[:, :2]
else:
    assert dType in ['MAG', 'GRAV', 'XYZ'], "dType must be 'MAG' or 'GRAV'"

# Downsample the survey using specified method
assert method[0] in ['radius', 'random'], "Downsample method should be 'radius' or 'random' "


def progress(iter, prog, final):
    """
    progress(iter,prog,final)

    Function measuring the progress of a process and print to screen the %.
    Useful to estimate the remaining runtime of a large problem.

    Created on Dec, 20th 2015

    @author: dominiquef
    """
    arg = np.floor(float(iter)/float(final)*10.)

    if arg > prog:

        print("Done " + str(arg*10) + " %")
        prog = arg

    return prog


if method[0] == 'radius':

    tree = cKDTree(locXYZ[:, :2])

    nstn = locXYZ.shape[0]
    # Initialize the filter
    indx = np.ones(nstn, dtype='bool')

    count = -1
    print("Begin filtering for radius= " + str(method[1]))

    for ii in range(nstn):

        if indx[ii]:

            ind = tree.query_ball_point(locXYZ[ii, :2], method[1])

            indx[ind] = False
            indx[ii] = True

        count = progress(ii, count, nstn)


elif method[0] == 'random':

    nD = int(locXYZ.shape[0]*method[1])
    print("nD ratio:" + str(nD) + '\\' + str(locXYZ.shape[0]))
    indx = np.random.randint(0, high=locXYZ.shape[0], size=nD)


# Create a new downsampled survey
if dType == 'MAG':

    rxLoc = PF.BaseGrav.RxObs(locXYZ[indx, :])
    srcField = PF.BaseMag.SrcField([rxLoc], param=survey.srcField.param)
    survey_dwnS = PF.BaseMag.LinearSurvey(srcField)
    survey_dwnS.dobs = survey.dobs[indx]
    survey_dwnS.std = survey.std[indx]

    Utils.io_utils.writeUBCmagneticsObservations(workDir + '\\' + dFileOut, survey_dwnS, survey_dwnS.dobs)

elif dType == 'GRAV':

    rxLoc = PF.BaseGrav.RxObs(locXYZ[indx, :])
    srcField = PF.BaseGrav.SrcField([rxLoc])
    survey_dwnS = PF.BaseGrav.LinearSurvey(srcField)
    survey_dwnS.dobs = survey.dobs[indx]
    survey_dwnS.std = survey.std[indx]

    Utils.io_utils.writeUBCgravityObservations(workDir + '\\' + dFileOut, survey_dwnS, survey_dwnS.dobs)

elif dType == 'XYZ':

    # vec = np.zeros(locXYZ.shape[0], dtype='bool')
    # vec[indx] = True
    # indx = np.all([vec, locXYZ[:,0] > 479000, locXYZ[:,1] > 6910000,
    #                locXYZ[:,0] < 670000, locXYZ[:,1] < 7009000], axis=0)
    survey_swnS = survey[indx, :]  # np.c_[survey[indx, :2],survey[indx, -1]]
    np.savetxt(workDir + '\\' + dFileOut, survey_swnS)
