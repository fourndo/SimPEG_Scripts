from SimPEG import Mesh, Utils
from scipy.spatial import cKDTree
import numpy as np
import sys


meshOctreeFile = sys.argv[3]
modelTensorFile = sys.argv[2]
meshTensorFile = sys.argv[1]

if len(sys.argv) > 4:
    ndv = np.asarray(sys.argv[4]).astype(np.float)
else:
    ndv = -99999

if len(sys.argv) > 5:
    vec = True
else:
    vec = False

# workDir = "C:/Users/DominiqueFournier/Dropbox/Projects/Felder/Modeling/"

# modelFile = ["MVI_S_TOT_l2222.amp", "MVI_S_TOT_l0222.amp"]

meshTensor = Mesh.TensorMesh.readUBC(meshTensorFile)


meshOctree = Mesh.TreeMesh.readUBC(meshOctreeFile)

if vec:
    vec_model = np.loadtxt(modelTensorFile)


    m1 = np.reshape(vec_model[:, 0], (meshTensor.nCz, meshTensor.nCx, meshTensor.nCy), order='F')
    m1 = m1[::-1, :, :]
    m1 = np.transpose(m1, (1, 2, 0))
    m1 = Utils.mkvc(m1)

    m2 = np.reshape(vec_model[:, 1], (meshTensor.nCz, meshTensor.nCx, meshTensor.nCy), order='F')
    m2 = m2[::-1, :, :]
    m2 = np.transpose(m2, (1, 2, 0))
    m2 = Utils.mkvc(m2)

    m3 = np.reshape(vec_model[:, 2], (meshTensor.nCz, meshTensor.nCx, meshTensor.nCy), order='F')
    m3 = m3[::-1, :, :]
    m3 = np.transpose(m3, (1, 2, 0))
    m3 = Utils.mkvc(m3)

else:
    model = meshTensor.readModelUBC(modelTensorFile)

# indAct = model != np.asarray(ndv)

tree = cKDTree(meshTensor.gridCC)

dd, ind = tree.query(meshOctree.gridCC)


if vec:
    mOut = np.ones((meshOctree.nC, 3))*ndv
    mOut[:, 0] = m1[ind]
    mOut[:, 1] = m2[ind]
    mOut[:, 2] = m3[ind]

else:
    mOut = np.ones(meshOctree.nC)*ndv
    mOut = model[ind]


# meshOctree.writeModelUBC(modelOctreeFile[:-4] + "_TENSOR" + modelOctreeFile[-4:], mOut)
if vec:
    Utils.io_utils.writeVectorUBC(
        meshOctree,
        modelTensorFile[:-4] + '_Octree' + modelTensorFile[-4:],
        mOut
    )

else:
    Mesh.TreeMesh.writeUBC(
          meshOctree, meshOctreeFile,
          models={modelTensorFile[:-4] + '_Octree' + modelTensorFile[-4:]: mOut}
        )

print("Export model file " + modelTensorFile[:-4] + "_Octree" + modelTensorFile[-4:] + " completed!" )

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
