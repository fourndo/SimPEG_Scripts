# -*- coding: utf-8 -*-
"""
Mag inversion over block.
Animation for SEG-Dallas
"""

from SimPEG import *
import SimPEG.PF as PF
import simpegCoordUtils as Utils
import matplotlib.pyplot as plt
import Mag as MAG
import simpegPF as PFlocal
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import animation
from JSAnimation import HTMLWriter
#%matplotlib notebook

# Add ploting function
# NEED TO HOOK IT UP WITH INVERSION CODE
fig = plt.figure(figsize=(10,5))
ax1 = plt.subplot(121, projection='3d')
pos = ax1.get_position()

ax2 =fig.add_axes([pos.x0 + 0.38, pos.y0+0.37,  pos.height*0.6, pos.height*0.6])

ax3 =fig.add_axes([pos.x0 + 0.5, pos.y0+0.05,  pos.height*0.3, pos.height*0.3])

ax4 = fig.add_axes([pos.x0 + 0.75, pos.y0+0.25,  pos.height*0.02, pos.height*0.4])

ax5 = fig.add_axes([pos.x0 , pos.y0+0.25,  pos.height*0.02, pos.height*0.4])

indz = 12
indy = 10

# Define the dimensions of the prism (m)
dx, dy, dz = 75., 2., 25.
# Set the depth of burial (m)
depth = 10.
pinc, pdec = 30., 90.
npts2D, xylim = 20., 40.
rx_h, View_elev, View_azim = 5., 15, -91
Einc, Edec, Bigrf = 45., 45., 50000.
x1, x2, y1, y2 = x1, x2, y1, y2 = -xylim, xylim, 0., 0.
comp = 'tf'
irt = 'total'
Q, rinc,rdec = 0., 60., 270.
susc = 0.25

vmin, vmax = 0., 0.015

ax1.axis('equal')
ax1.set_title('Forward Simulation')
# Define the problem interactively
p = MAG.definePrism()
p.dx, p.dy, p.dz, p.z0 = dx, dy, dz, -depth
p.pinc, p.pdec = pinc, pdec

srvy = PFlocal.survey()
srvy.rx_h, srvy.npts2D, srvy.xylim = rx_h, npts2D, xylim

# Create problem
prob = PFlocal.problem()
prob.prism = p
prob.survey = srvy

X, Y = np.meshgrid(prob.survey.xr, prob.survey.yr)
Z = np.ones(X.shape)*rx_h
x, y = MAG.linefun(x1, x2, y1, y2, prob.survey.npts2D)
xyz_line = np.c_[x, y, np.ones_like(x)*prob.survey.rx_h]

# Create a mesh
dx    = 5.

hxind = [(dx,5,-1.3), (dx, 10), (dx,5,1.3)]
hyind = [(dx,5,-1.3), (dx, 10), (dx,5,1.3)]
hzind = [(dx,5,-1.3),(dx, 10)]

mesh = Mesh.TensorMesh([hxind, hyind, hzind], x0='CCN')

# Get index of the center
midx = int(mesh.nCx/2)
midy = int(mesh.nCy/2)

model = np.zeros((mesh.nCx,mesh.nCy,mesh.nCz))


ax1.plot(x,y,xyz_line[:,2], 'w.', ms=3,lw=2)

#ax1.text(-1,0., -3.25, 'B-field', fontsize=14, color='k', horizontalalignment='left')

im1 = ax1.contourf(X,Y,X)
im2 = ax2.contourf(X,Y,X)
im3 = ax3.contourf(X,Y,X)
##im4 = ax2.plot(x,y)
im5 = ax1.text(0,0,0,'')
clim = np.asarray([-100,150])
def animate(ii):

    removePlt()
#ii=1
    if ii<18:
        dec = 90
        inc = 0. + ii*5.

    elif ii < 36:

        dec = 270.
        inc = 90. - (ii-18)*5.
        
    elif ii < 54:
        
        dec = 270.
        inc = 0.+ (ii-36)*5.
        
    else:
        
        dec = 90
        inc = 90. - (ii-54)*5.



    ax1.axis('equal')
    block_xyz = np.asarray([[-.2, -.2, .2, .2, 0],
                           [-.25, -.25, -.25, -.25, 0.5],
                           [-.2, .2, .2, -.2, 0]])*10.

    block_xyz[1][:] -=20.
    # rot = Utils.mkvc(Utils.dipazm_2_xyz(pinc, pdec))

    # xyz = Utils.rotatePointsFromNormals(block_xyz.T, np.r_[0., 1., 0.], rot,
    #                                     np.r_[p.xc, p.yc, p.zc])

    R = Utils.rotationMatrix(inc, dec)

    xyz = R.dot(block_xyz).T
    xyz[:,2] -= depth + dz/2.
    #print xyz
    # Face 1
    ax1.add_collection3d(Poly3DCollection([zip(xyz[:4, 0],
                                               xyz[:4, 1],
                                               xyz[:4, 2])], facecolors='b'))

    ax1.add_collection3d(Poly3DCollection([zip(xyz[[1, 2, 4], 0],
                                               xyz[[1, 2, 4], 1],
                                               xyz[[1, 2, 4], 2])], facecolors='b'))

    ax1.add_collection3d(Poly3DCollection([zip(xyz[[0, 1, 4], 0],
                                               xyz[[0, 1, 4], 1],
                                               xyz[[0, 1, 4], 2])], facecolors='b'))

    ax1.add_collection3d(Poly3DCollection([zip(xyz[[2, 3, 4], 0],
                                               xyz[[2, 3, 4], 1],
                                               xyz[[2, 3, 4], 2])], facecolors='b'))

    ax1.add_collection3d(Poly3DCollection([zip(xyz[[0, 3, 4], 0],
                                           xyz[[0, 3, 4], 1],
                                           xyz[[0, 3, 4], 2])], facecolors='b'))
    ax1.w_yaxis.set_ticklabels('')
    ax1.w_yaxis.set_label_text('')
    ax1.w_zaxis.set_ticklabels('')
    ax1.w_zaxis.set_label_text('')
#    block_xyz[1][:] +=20.
#    # rot = Utils.mkvc(Utils.dipazm_2_xyz(pinc, pdec))
#
#    # xyz = Utils.rotatePointsFromNormals(block_xyz.T, np.r_[0., 1., 0.], rot,
#    #                                     np.r_[p.xc, p.yc, p.zc])
#
#    R = Utils.rotationMatrix(rinc, rdec)
#
#    xyz = R.dot(block_xyz).T
#    xyz[:,2] -= depth + dz/2.
#
#    #print xyz
#    # Face 1
#    ax1.add_collection3d(Poly3DCollection([zip(xyz[:4, 0],
#                                               xyz[:4, 1],
#                                               xyz[:4, 2])], facecolors='y'))
#
#    ax1.add_collection3d(Poly3DCollection([zip(xyz[[1, 2, 4], 0],
#                                               xyz[[1, 2, 4], 1],
#                                               xyz[[1, 2, 4], 2])], facecolors='y'))
#
#    ax1.add_collection3d(Poly3DCollection([zip(xyz[[0, 1, 4], 0],
#                                               xyz[[0, 1, 4], 1],
#                                               xyz[[0, 1, 4], 2])], facecolors='y'))
#
#    ax1.add_collection3d(Poly3DCollection([zip(xyz[[2, 3, 4], 0],
#                                               xyz[[2, 3, 4], 1],
#                                               xyz[[2, 3, 4], 2])], facecolors='y'))
#
#    ax1.add_collection3d(Poly3DCollection([zip(xyz[[0, 3, 4], 0],
#                                           xyz[[0, 3, 4], 1],
#                                           xyz[[0, 3, 4], 2])], facecolors='y'))

    MAG.plotObj3D(p, rx_h, View_elev, View_azim, npts2D, xylim, profile="X", fig= fig, axs = ax1, plotSurvey=False)
    # Create problem
    prob = PFlocal.problem()
    prob.prism = p
    prob.survey = srvy

    prob.Bdec, prob.Binc, prob.Bigrf = dec, inc, Bigrf
    prob.Q, prob.rinc, prob.rdec = Q, rinc, rdec
    prob.uType, prob.mType = comp, 'total'
    prob.susc = susc

    # Compute fields from prism
    b_ind, b_rem = prob.fields()

    if irt == 'total':
        out = b_ind + b_rem

    elif irt == 'induced':
        out = b_ind

    else:
        out = b_rem

    #out = plogMagSurvey2D(prob, susc, Einc, Edec, Bigrf, x1, y1, x2, y2, comp, irt,  Q, rinc, rdec, fig=fig, axs1=ax2, axs2=ax3)

    #dat = axs1.contourf(X,Y, np.reshape(out, (X.shape)).T
    global im1
    im1 = ax1.contourf(X,Y,np.reshape(out, (X.shape)).T,20,zdir='z',offset=rx_h+5., clim=clim, vmin=clim[0],vmax=clim[1], cmap = 'RdBu_r')

    ax5 = fig.add_axes([pos.x0 , pos.y0+0.25,  pos.height*0.02, pos.height*0.4])
    cb = plt.colorbar(im1,cax=ax5, orientation="vertical", ax = ax1, ticks=np.linspace(im1.vmin,im1.vmax, 4), format="${%.0f}$")
    cb.set_label("$B^{TMI}\;(nT)$",size=12)
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.yaxis.set_label_position('left')

    global im5
    im5 = ax1.text(0,0,-60,'$B_0, I: ' + str(inc) + '^\circ, D: ' + str(dec) + '^\circ$', horizontalalignment='center')


    H0 = (Bigrf,inc,dec)


    actv = np.ones(mesh.nC)==1
    # Create active map to go from reduce space to full
    actvMap = Maps.InjectActiveCells(mesh, actv, -100)
    nC = len(actv)

    # Create a MAGsurvey
    rxLoc = np.c_[Utils.mkvc(X), Utils.mkvc(Y), Utils.mkvc(Z)]
    rxLoc = PF.BaseMag.RxObs(rxLoc)
    srcField = PF.BaseMag.SrcField([rxLoc],param = H0)
    survey = PF.BaseMag.LinearSurvey(srcField)

    # We can now create a susceptibility model and generate data
    # Lets start with a simple block in half-space
#    model = np.zeros((mesh.nCx,mesh.nCy,mesh.nCz))
#    model[(midx-2):(midx+2),(midy-2):(midy+2),-6:-2] = 0.02
#    model = mkvc(model)
#    model = model[actv]

    # Create active map to go from reduce set to full
    actvMap = Maps.InjectActiveCells(mesh, actv, -100)

    # Creat reduced identity map
    idenMap = Maps.IdentityMap(nP = nC)

    # Create the forward model operator
    probinv = PF.Magnetics.MagneticIntegral(mesh, mapping = idenMap, actInd = actv)

    # Pair the survey and problem
    survey.pair(probinv)

    # Compute linear forward operator and compute some data
#    d = probinv.fields(model)

    # Plot the model
#    m_true = actvMap * model
#    m_true[m_true==-100] = np.nan
    #plt.figure()
    #ax = plt.subplot(212)
    #mesh.plotSlice(m_true, ax = ax, normal = 'Y', ind=midy, grid=True, clim = (0., model.max()/3.), pcolorOpts={'cmap':'viridis'})
    #plt.title('A simple block model.')
    #plt.xlabel('x'); plt.ylabel('z')
    #plt.gca().set_aspect('equal', adjustable='box')

    # We can now generate data
    data = out + np.random.randn(len(out)) # We add some random Gaussian noise (1nT)
    wd = np.ones(len(data))*1. # Assign flat uncertainties

    # Create distance weights from our linera forward operator
    wr = np.sum(probinv.G**2.,axis=0)**0.5
    wr = ( wr/np.max(wr) )

    #survey.makeSyntheticData(data, std=0.01)
    survey.dobs= data
    survey.std = wd
    survey.mtrue = model

    # Create a regularization
    reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
    reg.cell_weights = wr

    dmis = DataMisfit.l2_DataMisfit(survey)
    dmis.Wd = 1/wd

    # Add directives to the inversion
    opt = Optimization.ProjectedGNCG(maxIter=100 ,lower=0.,upper=1., maxIterLS = 20, maxIterCG= 10, tolCG = 1e-3)
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    betaest = Directives.BetaEstimate_ByEig()

    # Here is where the norms are applied
    # Use pick a treshold parameter empirically based on the distribution of model
    # parameters (run last cell to see the histogram before and after IRLS)
    IRLS = Directives.Update_IRLS( norms=([2,2,2,2]),  eps=(2e-3,2e-3), f_min_change = 1e-3, minGNiter=3,beta_tol=1e-2)
    update_Jacobi = Directives.Update_lin_PreCond()
    inv = Inversion.BaseInversion(invProb, directiveList=[IRLS,betaest,update_Jacobi])

    m0 = np.ones(nC)*1e-4

    mrec = inv.run(m0)

    # Here is the recovered susceptibility model
    ypanel = midx
    zpanel = -4
    m_l2 = actvMap * reg.l2model
    m_l2[m_l2==-100] = np.nan

    m_lp = actvMap * mrec
    m_lp[m_lp==-100] = np.nan

#    m_true = actvMap * model
#    m_true[m_true==-100] = np.nan

    #Plot L2 model
    #global im2

    xx, zz = mesh.gridCC[:,0].reshape(mesh.vnC, order="F"), mesh.gridCC[:,2].reshape(mesh.vnC, order="F")
    yy = mesh.gridCC[:,1].reshape(mesh.vnC, order="F")

    temp = m_lp.reshape(mesh.vnC, order='F')
    ptemp = temp[:,:,indz].T
    #ptemp = ma.array(ptemp ,mask=np.isnan(ptemp))
    global im2
    im2 = ax2.contourf(xx[:,:,indz].T,yy[:,:,indz].T,ptemp,20, vmin = vmin, vmax= vmax, clim=[vmin,vmax])
    ax2.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCy[indy],mesh.vectorCCy[indy]]),color='w')
    ax2.set_aspect('equal')
    ax2.xaxis.set_visible(False)
    ax2.set_xlim(-60,60)
    ax2.set_ylim(-60,60)
    ax2.set_title('Induced Model')
    ax2.set_ylabel('Northing (m)',size=14)


    ptemp = temp[:,indy,:].T
    global im3
    im3 = ax3.contourf(xx[:,indy,:].T,zz[:,indy,:].T,ptemp,20, vmin = vmin, vmax= vmax, clim=[vmin,vmax])
    ax3.set_aspect('equal')
    ax3.set_xlim(-60,60)
    ax3.set_ylim(-60,0)
    ax3.set_title('EW Section')
    ax3.set_xlabel('Easting (m)',size=14)
    ax3.set_ylabel('Elevation (m)',size=14)


    ax4 = fig.add_axes([pos.x0 + 0.75, pos.y0+0.25,  pos.height*0.02, pos.height*0.4])
    cb = plt.colorbar(im3,cax=ax4, orientation="vertical", ax = ax1, ticks=np.linspace(im3.vmin,im3.vmax, 4), format="${%.3f}$")
    cb.set_label("Susceptibility (SI)",size=12)
#ax2.axis('equal')

#im2 = mesh.plotSlice(m_l2, ax = ax2, normal = 'Z', ind=zpanel, grid=True, clim = (0., model.max()/3.),pcolorOpts={'cmap':'viridis'})
##plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCy[ypanel],mesh.vectorCCy[ypanel]]),color='w')
#plt.title('Plan l2-model.')
#ax2.set_aspect('equal')
#plt.ylabel('y')
#ax2.xaxis.set_visible(False)
#ax2.set_xlim(-60,60)
#ax2.set_ylim(-60,60)
#ax2.axis('equal')

# Vertica section
#global im3
#im3 = mesh.plotSlice(m_l2, ax = ax3, normal = 'Y', ind=midx, grid=True, clim = (0., model.max()/3.),pcolorOpts={'cmap':'viridis'})
##plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([mesh.vectorCCz[zpanel],mesh.vectorCCz[zpanel]]),color='w')
##plt.plot(([mesh.vectorCCx[0],mesh.vectorCCx[-1]]),([Z.min(),Z.max()]),color='k')
#plt.title('E-W l2-model.')
#
#plt.xlabel('x')
#plt.ylabel('z')
#ax3.set_xlim(-60,60)
#ax3.set_ylim(-80,0)
#ax3.set_aspect('equal')
#ax3.axis('equal')
#
def removePlt():
    #global im1
    #im1.remove()

    global im1
    for coll in im1.collections:
        coll.remove()

    for cc in range(6):
        for coll in ax1.collections:
            ax1.collections.remove(coll)


    global im2
    for coll in im2.collections:
        coll.remove()


    global im3
    for coll in im3.collections:
        coll.remove()

#    global im4
#    im4.pop(0).remove()
#
    global im5
    im5.remove()

anim = animation.FuncAnimation(fig, animate,
                               frames=36, interval=200,repeat=False)

anim.save('animationV2.mp4', writer='ffmpeg')



