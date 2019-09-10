# -*- coding: utf-8 -*-
"""
Created on Wed May  9 13:20:56 2018

@author: craigm


xyzLocs are the points of topo and obs
h is the core discretization. right now it has to be [x, x, x], (cubes)
padDist is how far out you want to go for the padding
padCore is the number of cells in the three finest octree levels
beyond the xyzLocs
gridLoc changes where the top of topography is, either in the center ('CC') or the top ('N') of the octree mesh
the total mesh width and depth dimensions should be 2^n like, 2,4,8,16,32,64,128,256

"""
from SimPEG import Mesh, Utils
import SimPEG.PF as PF
import numpy as np
import os
from scipy.spatial import cKDTree
from SimPEG.Utils import mkvc

work_dir = ".\\Assets\\"
out_dir = "Assets\\"
input_file = "SimPEG_MAG.inp"


padLen = 300  # Padding distance around the convex haul of the data
octreeObs = [0, 5, 10]  # Octree levels below data points [n1*dz, n2*dz**2, ...]
octreeTopo = [0, 1, 1]   # Octree levels below topography [n1*dz, n2*dz**2, ...]
octreeTopoFine = [1, 0, 0]
distMax = 100  # Maximum distance (m) outside convex haul to discretize

# %%
# Read in the input file which included all parameters at once
# (mesh, topo, model, survey, inv param, etc.)

def meshBuilder(xyz, h, padDist, meshGlobal=None,
                expFact=1.3,
                meshType='TENSOR',
                verticalAlignment='top'):
    """
        Function to quickly generate a Tensor mesh
        given a cloud of xyz points, finest core cell size
        and padding distance.
        If a meshGlobal is provided, the core cells will be centered
        on the underlaying mesh to reduce interpolation errors.

        :param numpy.ndarray xyz: n x 3 array of locations [x, y, z]
        :param numpy.ndarray h: 1 x 3 cell size for the core mesh
        :param numpy.ndarray padDist: 2 x 3 padding distances [W,E,S,N,Down,Up]
        [OPTIONAL]
        :param numpy.ndarray padCore: Number of core cells around the xyz locs
        :object SimPEG.Mesh: Base mesh used to shift the new mesh for overlap
        :param float expFact: Expension factor for padding cells [1.3]
        :param string meshType: Specify output mesh type: "TensorMesh"

        RETURNS:
        :object SimPEG.Mesh: Mesh object

    """

    assert meshType in ['TENSOR', 'TREE'], ('Revise meshType. Only ' +
                                            ' TENSOR | TREE mesh ' +
                                            'are implemented')

    # Get extent of points
    limx = np.r_[xyz[:, 0].max(), xyz[:, 0].min()]
    limy = np.r_[xyz[:, 1].max(), xyz[:, 1].min()]
    limz = np.r_[xyz[:, 2].max(), xyz[:, 2].min()]

    # Get center of the mesh
    midX = np.mean(limx)
    midY = np.mean(limy)

    if verticalAlignment == 'center':
        midZ = np.mean(limz)
    else:
        midZ = limz[0]

    nCx = int(limx[0]-limx[1]) / h[0]
    nCy = int(limy[0]-limy[1]) / h[1]
    nCz = int(np.max([
            limz[0]-limz[1],
            int(np.min(np.r_[nCx*h[0], nCy*h[1]])/3)
            ]) / h[2])

    if meshType == 'TENSOR':
        # Make sure the core has odd number of cells for centereing
        # on global mesh
        if meshGlobal is not None:
            nCx += 1 - int(nCx % 2)
            nCy += 1 - int(nCy % 2)
            nCz += 1 - int(nCz % 2)

        # Figure out paddings
        def expand(dx, pad):
            L = 0
            nC = 0
            while L < pad:
                nC += 1
                L = np.sum(dx * expFact**(np.asarray(range(nC))+1))

            return nC

        # Figure number of padding cells required to fill the space
        npadEast = expand(h[0], padDist[0, 0])
        npadWest = expand(h[0], padDist[0, 1])
        npadSouth = expand(h[1], padDist[1, 0])
        npadNorth = expand(h[1], padDist[1, 1])
        npadDown = expand(h[2], padDist[2, 0])
        npadUp = expand(h[2], padDist[2, 1])

        # Create discretization
        hx = [(h[0], npadWest, -expFact),
              (h[0], nCx),
              (h[0], npadEast, expFact)]
        hy = [(h[1], npadSouth, -expFact),
              (h[1], nCy), (h[1],
              npadNorth, expFact)]
        hz = [(h[2], npadDown, -expFact),
              (h[2], nCz),
              (h[2], npadUp, expFact)]

        # Create mesh
        mesh = Mesh.TensorMesh([hx, hy, hz], 'CC0')

        # Re-set the mesh at the center of input locations
        # Set origin
        if verticalAlignment == 'center':
            mesh.x0 = [midX-np.sum(mesh.hx)/2., midY-np.sum(mesh.hy)/2., midZ - np.sum(mesh.hz)/2.]
        elif verticalAlignment == 'top':
            mesh.x0 = [midX-np.sum(mesh.hx)/2., midY-np.sum(mesh.hy)/2., midZ - np.sum(mesh.hz)]
        else:
            assert NotImplementedError("verticalAlignment must be 'center' | 'top'")

    elif meshType == 'TREE':

        # Figure out full extent required from input
        extent = np.max(np.r_[nCx * h[0] + padDist[0, :].sum(),
                              nCy * h[1] + padDist[1, :].sum(),
                              nCz * h[2] + padDist[2, :].sum()])

        maxLevel = int(np.log2(extent/h[0]))+1

        # Number of cells at the small octree level
        # equal in 3D
        nCx, nCy, nCz = 2**(maxLevel), 2**(maxLevel), 2**(maxLevel)

        # nCy = 2**(int(np.log2(extent/h[1]))+1)
        # nCz = 2**(int(np.log2(extent/h[2]))+1)

        # Define the mesh and origin
        mesh = Mesh.TreeMesh([np.ones(nCx)*h[0],
                              np.ones(nCx)*h[1],
                              np.ones(nCx)*h[2]])

        # Shift mesh if global mesh is used
        center = np.r_[midX, midY, midZ]
        if meshGlobal is not None:

            tree = cKDTree(meshGlobal.gridCC)
            _, ind = tree.query(center, k=1)
            center = meshGlobal.gridCC[ind, :]

        # Set origin
        if verticalAlignment == 'center':
            mesh.x0 = np.r_[center[0] - (nCx-1)*h[0]/2., center[1] - (nCy-1)*h[1]/2., center[2] - (nCz-1)*h[2]/2.]
        elif verticalAlignment == 'top':
            mesh.x0 = np.r_[center[0] - (nCx-1)*h[0]/2., center[1] - (nCy-1)*h[1]/2., center[2] - (nCz-1)*h[2]]
        else:
            assert NotImplementedError("verticalAlignment must be 'center' | 'top'")

    return mesh


def refineTree(
            mesh, xyz,
            finalize=False, dtype="point",
            nCpad=[1, 1, 1], distMax=200,
            padRatio=2):

    maxLevel = int(np.log2(mesh.hx.shape[0]))

    if dtype == "point":

        mesh.insert_cells(xyz, np.ones(xyz.shape[0])*maxLevel, finalize=False)

        stencil = np.r_[
                np.ones(nCpad[0]),
                np.ones(nCpad[1])*2,
                np.ones(nCpad[2])*3
            ]

        # Reflect in the opposite direction
        vec = np.r_[stencil[::-1], 1, stencil]
        vecX, vecY, vecZ = np.meshgrid(vec, vec, vec)
        gridLevel = np.maximum(np.maximum(np.abs(vecX),
                               np.abs(vecY)), np.abs(vecZ))
        gridLevel = np.kron(np.ones(xyz.shape[0]), mkvc(gridLevel))

        # Grid the coordinates
        vec = np.r_[-np.cumsum(stencil)[::-1], 0, np.cumsum(stencil)]
        vecX, vecY, vecZ = np.meshgrid(vec, vec, vec)
        offset = np.c_[
            mkvc(np.sign(vecX)*np.abs(vecX) * mesh.hx.min()),
            mkvc(np.sign(vecY)*np.abs(vecY) * mesh.hy.min()),
            mkvc(np.sign(vecZ)*np.abs(vecZ) * mesh.hz.min())
        ]

        # Replicate the point locations in each offseted grid points
        newLoc = (
            np.kron(xyz, np.ones((offset.shape[0], 1))) +
            np.kron(np.ones((xyz.shape[0], 1)), offset)
        )

        mesh.insert_cells(
            newLoc, maxLevel-mkvc(gridLevel)+1, finalize=finalize
        )

    elif dtype == 'surface':

        # Get extent of points
        limx = np.r_[xyz[:, 0].max(), xyz[:, 0].min()]
        limy = np.r_[xyz[:, 1].max(), xyz[:, 1].min()]

        # Get center of the mesh
        midX = np.mean(limx)
        midY = np.mean(limy)

        dx = mesh.hx.min()
        dy = mesh.hy.min()

        nCx = int((limx[0]-limx[1]) / dx)
        nCy = int((limy[0]-limy[1]) / dy)

        tree = cKDTree(xyz[:, :2])
        # xi = _ndim_coords_from_arrays((gridCC[:,0], gridCC[:,1]), ndim=2)

        # Increment the vertical offset
        zOffset = 0
        depth = 0

        # Compute maximum depth of refinement
        dz = np.repeat(mesh.hz.min() * 2**np.arange(len(nCpad)), np.r_[nCpad])
        depth = dz.sum()

        # Cycle through the Tree levels backward
        for ii in range(len(nCpad)-1, -1, -1):
            dz = mesh.hz.min() * 2**ii

            # Increase the horizontal extent of the surface
            # as a function of Tree level
            # r = ((CCx-midX)**2. + (CCy-midY)**2.)**0.5
            # expFact = (r.max() + 2*mesh.hx.min()*2**ii)/r.max()
            # Create a grid at the octree level in xy
            CCx, CCy = np.meshgrid(
                np.linspace(
                    limx[1] - depth/padRatio, limx[0] + depth/padRatio, nCx
                    ),
                np.linspace(
                    limy[1] - depth/padRatio, limy[0] + depth/padRatio, nCy
                    )
            )

            dists, indexes = tree.query(np.c_[mkvc(CCx), mkvc(CCy)])

            # Copy original result but mask missing values with NaNs
            maskRadius = dists < (distMax + depth/padRatio)

            # Only keep points inside the convex hull
            x, y, z = mkvc(CCx)[maskRadius], mkvc(CCy)[maskRadius], xyz[indexes[maskRadius],2]
            zOffset = 0
            while zOffset < depth:

                mesh.insert_cells(
                    np.c_[x, y, z-zOffset], np.ones_like(z)*maxLevel-ii,
                    finalize=False
                )

                zOffset += dz

            depth -= dz * nCpad[ii]

        if finalize:
            mesh.finalize()

    else:
        NotImplementedError("Only dtype= 'surface' | 'points' has been implemented")

    return mesh


driver = PF.MagneticsDriver.MagneticsDriver_Inv(work_dir + input_file)


# Access the mesh and survey information
meshInput = driver.mesh
survey = driver.survey
actv = driver.activeCells

# Just get the xyz locations
xyzLocs = survey.srcField.rxList[0].locs.copy()

topo = None
if driver.topofile is not None:
    topo = np.genfromtxt(driver.basePath + driver.topofile,
                         skip_header=1)


# Create an additional topo layer point near the
# the data for higher discretization
tree = cKDTree(xyzLocs[:, :2])  # cKDTree finds the nearest neighbours
r, ii = tree.query(topo[:, :2])

# Copy original result but mask missing values with NaNs
maskRadius = r < distMax
topoFine = topo[ii, :]

if isinstance(meshInput, Mesh.TensorMesh):
    # Define an octree mesh based on the provided tensor
    h = np.r_[meshInput.hx.min(), meshInput.hy.min(), meshInput.hz.min()]
    padDist = np.r_[np.c_[padLen, padLen], np.c_[padLen, padLen], np.c_[padLen, 0]]

    print("Creating TreeMesh. Please standby...")
    mesh = meshBuilder(
        topo, h, padDist,
        meshGlobal=meshInput,
        meshType='TREE',
        verticalAlignment='center')

    # Large topo discretize
    mesh = refineTree(
        mesh, topo, dtype='surface',
        nCpad=octreeTopo, finalize=False)

    # Topo fine
    mesh = refineTree(
        mesh, topo, dtype='surface',
        distMax=distMax,
        nCpad=octreeTopoFine, finalize=False)

    # Obs discretize, and finalize
    mesh = refineTree(
        mesh, xyzLocs, dtype='surface',
        distMax=distMax,
        nCpad=octreeObs, finalize=True)


else:
    mesh = Mesh.TreeMesh.readUBC(driver.basePath + driver.mshfile)

if driver.topofile is not None:
    # Find the active cells
    actv = Utils.surface2ind_topo(mesh, topo)
else:
    actv = mesh.gridCC[:, 2] <= meshInput.vectorNz[-1]

nC = int(np.sum(actv))

# write the mesh and a file of active cells
Mesh.TreeMesh.writeUBC(
    mesh,
    out_dir + 'OctreeMesh.msh',
    models={out_dir + 'ActiveOctree.dat': actv}
)
