"""
modelUBC2xyz.py mesh model no-data-value

Python routine for the conversion of a UBC model to xyz file format.
"""

import numpy as np
import sys
from SimPEG import Mesh

# Get mesh and model files from input
mesh_file = sys.argv[1]
model_file = sys.argv[2]
ndv = np.asarray(sys.argv[3]).astype(np.float)


mesh = Mesh.TreeMesh.readUBC(mesh_file)
model = mesh.readModelUBC(model_file)

indAct = model != np.asarray(ndv)

out = np.c_[mesh.gridCC[indAct, :], model[indAct]]

np.savetxt(model_file[:-3] + "XYZ.xyz", out)

print("Model file " + model_file[:-3] + "XYZ.xyz completed!" )

