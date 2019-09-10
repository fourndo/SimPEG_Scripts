from SimPEG import Mesh, Utils
from scipy.spatial import cKDTree
import numpy as np
import sys


meshOctreeFile = sys.argv[1]
modelOctreeFile = sys.argv[2]
meshTensorFile = sys.argv[3]
ndv = np.asarray(sys.argv[4]).astype(np.float)

# workDir = "C:/Users/DominiqueFournier/Dropbox/Projects/Felder/Modeling/"

# modelFile = ["MVI_S_TOT_l2222.amp", "MVI_S_TOT_l0222.amp"]
meshTensor = Mesh.TensorMesh.readUBC(meshTensorFile)


meshOctree = Mesh.TreeMesh.readUBC(meshOctreeFile)

model = meshOctree.readModelUBC(modelOctreeFile)

# indAct = model != np.asarray(ndv)

tree = cKDTree(meshOctree.gridCC)

dd, ind = tree.query(meshTensor.gridCC)


mOut = np.ones(meshTensor.nC)*ndv
mOut = model[ind]


meshTensor.writeModelUBC(modelOctreeFile[:-4] + "_TENSOR" + modelOctreeFile[-4:], mOut)

print("Export model file " + modelOctreeFile[:-4] + "_TENSOR" + modelOctreeFile[-4:] + " completed!" )

# Model lp out
# vec_xyz = Utils.matutils.atp2xyz(
#     mrec_MVI_S.reshape((nC, 3), order='F')).reshape((nC, 3), order='F')
# vec_x = actvMap * vec_xyz[:, 0]
# vec_y = actvMap * vec_xyz[:, 1]
# vec_z = actvMap * vec_xyz[:, 2]

# vec_xyzTensor = np.zeros((meshTensor.nC, 3))
# vec_xyzTensor = np.c_[vec_x[ind], vec_y[ind], vec_z[ind]]

# Utils.io_utils.writeVectorUBC(
#     meshTensor, work_dir + 'MVI_S_TensorLp.fld', vec_xyzTensor)
# amp = np.sum(vec_xyzTensor**2., axis=1)**0.5
# meshTensor.writeModelUBC(workDir + 'MVI_S_TensorLp.amp', amp)

# # Model l2 out
# vec_xyz = Utils.matutils.atp2xyz(
#     invProb.l2model.reshape((nC, 3), order='F')).reshape((nC, 3), order='F')
# vec_x = actvMap * vec_xyz[:, 0]
# vec_y = actvMap * vec_xyz[:, 1]
# vec_z = actvMap * vec_xyz[:, 2]

# vec_xyzTensor = np.zeros((meshTensor.nC, 3))
# vec_xyzTensor = np.c_[vec_x[ind], vec_y[ind], vec_z[ind]]

# Utils.io_utils.writeVectorUBC(
#     meshTensor, work_dir + 'MVI_S_TensorL2.fld', vec_xyzTensor)

# amp = np.sum(vec_xyzTensor**2., axis=1)**0.5
# meshTensor.writeModelUBC(workDir + 'MVI_S_TensorL2.amp', amp)
