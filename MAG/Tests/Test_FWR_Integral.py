# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 09:47:38 2016


@author: dominiquef
"""

from SimPEG import Mesh, Utils, np, PF, Maps, Problem, Survey, mkvc
import matplotlib.pyplot as plt

# Define inducing field and sphere parameters
H0 = (50000., 60., 270.)
b0 = PF.MagAnalytics.IDTtoxyz(-H0[1], H0[2], H0[0])
rad = 2.
chi = 0.01

# Define a mesh
cs = 0.2
hxind = [(cs,10,1.3),(cs, 21),(cs,10,1.3)]
hyind = [(cs,10,1.3),(cs, 21),(cs,10,1.3)]
hzind = [(cs,10,1.3),(cs, 21),(cs,10,1.3)]
mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCC')

# Get cells inside the sphere
sph_ind = PF.MagAnalytics.spheremodel(mesh, 0., 0., 0., rad)

# Adjust susceptibility for volume difference
Vratio = (4./3.*np.pi*rad**3.) / (np.sum(sph_ind)*cs**3.)
model = np.zeros(mesh.nC)
model[sph_ind] = chi*Vratio
m = model[sph_ind]

# Creat reduced identity map for Linear Pproblem
idenMap = Maps.IdentityMap(nP=int(sum(sph_ind)))

# Create plane of observations
xr = np.linspace(-10, 10, 21)
yr = np.linspace(-10, 10, 21)
X, Y = np.meshgrid(xr, yr)

# Move obs plane 2 radius away from sphere
Z = np.ones((xr.size, yr.size))*2.*rad
locXyz = np.c_[Utils.mkvc(X), Utils.mkvc(Y), Utils.mkvc(Z)]
rxLoc = PF.BaseMag.RxObs(locXyz)
srcField = PF.BaseMag.SrcField([rxLoc], param=H0)
survey = PF.BaseMag.LinearSurvey(srcField)

prob_xyz = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap,
                                              actInd=sph_ind,
                                              forwardOnly=True,
                                              rtype='xyz')

prob_tmi = PF.Magnetics.MagneticIntegral(mesh, chiMap=idenMap,
                                              actInd=sph_ind,
                                              forwardOnly=True,
                                              rtype='tmi')

# Compute 3-component mag data
survey.pair(prob_xyz)
d = prob_xyz.fields(m)

ndata = locXyz.shape[0]
dbx = d[0:ndata]
dby = d[ndata:2*ndata]
dbz = d[2*ndata:]

# Compute tmi mag data
survey.pair(prob_tmi)
dtmi = prob_tmi.fields(m)

# Compute analytical response from a magnetized sphere
bxa, bya, bza = PF.MagAnalytics.MagSphereFreeSpace(locXyz[:, 0],
                                                   locXyz[:, 1],
                                                   locXyz[:, 2],
                                                   rad, 0, 0, 0,
                                                   chi, b0)

# Projection matrix
Ptmi = mkvc(b0)/np.sqrt(np.sum(b0**2.))

btmi = mkvc(Ptmi.dot(np.vstack((bxa, bya, bza))))

err_xyz = (np.linalg.norm(d-np.r_[bxa, bya, bza]) /
           np.linalg.norm(np.r_[bxa, bya, bza]))

err_tmi = np.linalg.norm(dtmi-btmi)/np.linalg.norm(btmi)

#fig = plt.figure()
#axs = plt.subplot(111)
#mesh.plotSlice(model, normal='Z', ind = mesh.nCz/2, ax= axs)
#axs.set_aspect('equal')

#PF.Magnetics.plot_obs_2D(locXyz,dtmi)

#%% Repeat using PDE solve
m_mu = PF.BaseMag.BaseMagMap(mesh)
prob = PF.Magnetics.Problem3D_DiffSecondary(mesh, muMap=m_mu)

survey = PF.BaseMag.BaseMagSurvey()
survey.setBackgroundField(H0[1], H0[2], H0[0])
survey.rxLoc = locXyz

prob.pair(survey)
u = prob.fields(model)
dtmi_PDE = survey.projectFields(u)

PF.Magnetics.plot_obs_2D(locXyz,dtmi_PDE)
PF.Magnetics.plot_obs_2D(locXyz,btmi-dtmi_PDE,varstr='Residual between integral and PDE')