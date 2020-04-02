from SimPEG import Mesh, Utils
from scipy.spatial import cKDTree
import numpy as np
import sys


mesh_file_in = sys.argv[1]
model_file_in = sys.argv[2]
mesh_file_out = sys.argv[3]

if len(sys.argv) > 4:
    ndv = np.asarray(sys.argv[4]).astype(np.float)
else:
    ndv = -99999

model = np.loadtxt(model_file_in)

# Determine if the mesh is tensor or tree
fid = open(mesh_file_in, 'r')
for ii in range(6):
    line = fid.readline()
fid.close()

if line:
    input_mesh = Mesh.TreeMesh.readUBC(mesh_file_in)

    for ii range(model.shape[1]):

        m1 = np.reshape(vec_model[:, ii], (input_mesh.nCz, input_mesh.nCx, input_mesh.nCy), order='F')
        m1 = m1[::-1, :, :]
        m1 = np.transpose(m1, (1, 2, 0))
        model[:, ii] = Utils.mkvc(m1)

else:
    input_mesh = Mesh.TensorMesh.readUBC(mesh_file_in)
    for ii range(model.shape[1]):

        m1 = np.reshape(vec_model[:, ii], (input_mesh.nCz, input_mesh.nCx, input_mesh.nCy), order='F')
        m1 = m1[::-1, :, :]
        m1 = np.transpose(m1, (1, 2, 0))
        model[:, ii] = Utils.mkvc(m1)

fid = open(mesh_file_out, 'r')
for ii in range(6):
    line = fid.readline()
fid.close()

if line:
    input_mesh = Mesh.TreeMesh.readUBC(mesh_file_out)
else:
    input_mesh = Mesh.TensorMesh.readUBC(mesh_file_out)





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
