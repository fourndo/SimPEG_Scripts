from SimPEG import Mesh, Utils
from scipy.spatial import cKDTree
import numpy as np
import sys


#meshOctreeFile = sys.argv[1]
#modelOctreeFile = sys.argv[2]
#meshTensorFile = sys.argv[3]
#ndv = np.asarray(sys.argv[4]).astype(np.float)

meshOctreeFile = r"E:\Inversions\EQS_Testing\Kevitsa\ForwardModel_v3_core\OctreeMesh.msh"
#modelFile = r"E:\Inversions\EQS_Testing\Kevitsa\ForwardModel_fix2\GeoModelOCtree.mod"
modelOctreeFile = r"E:\Inversions\EQS_Testing\Kevitsa\ForwardModel_v3_core\True_Amp.sus"
#modelOctreeFile = r"E:\Inversions\EQS_Testing\Kevitsa\ForwardModel_v3_core\True_vector.fld"
activeOctreeFile = r"E:\Inversions\EQS_Testing\Kevitsa\ForwardModel_v3_core\Active.act"
meshTensorFile = r"E:\Inversions\EQS_Testing\Kevitsa\ForwardModel_fix2\Forward_AMP_RTP_30m.msh"

method = 'exact' # Or ''nearest'
vec = False

# workDir = "C:/Users/DominiqueFournier/Dropbox/Projects/Felder/Modeling/"

# modelFile = ["MVI_S_TOT_l2222.amp", "MVI_S_TOT_l0222.amp"]
meshTensor = Mesh.TensorMesh.readUBC(meshTensorFile)

meshOctree = Mesh.TreeMesh.readUBC(meshOctreeFile)

if len(sys.argv) > 4:
    ndv = np.asarray(sys.argv[4]).astype(np.float)
else:
    if vec:
        ndv = 0.0
    else:
        ndv = -100

if vec:
    model = Utils.io_utils.readVectorUBC(meshOctree, modelOctreeFile)
else:
    model = meshOctree.readModelUBC(modelOctreeFile)

active = meshOctree.readModelUBC(activeOctreeFile)

# indAct = model != np.asarray(ndv)

if method == 'exact':
    # All tensor cell centers contained within an octree cell volume will be
    # assigned the properties of that octree cell.
    cell_indices = meshOctree.point2index(meshTensor.gridCC)
    
else:
    # Each tensor cell will take properties from nearest octree cell center
    # even if the tensor cell is contained within an adjacent larger octree cell whose
    # center is further away.
    tree = cKDTree(meshOctree.gridCC)
    
    _, cell_indices = tree.query(meshTensor.gridCC, p=3, n_jobs=7)
    

activeOut = active[cell_indices]
if vec:
    mOut = model[cell_indices,:]
    mOut[activeOut <= 0, :] = ndv
else:
    mOut = model[cell_indices]
    mOut[activeOut <= 0] = ndv

if vec:
    Utils.io_utils.writeVectorUBC(
        meshTensor,
        modelOctreeFile[:-4] + '_TENSOR' + modelOctreeFile[-4:],
        mOut
    )

else:
    meshTensor.writeModelUBC(modelOctreeFile[:-4] + "_TENSOR" + modelOctreeFile[-4:], mOut)
    meshTensor.writeModelUBC(activeOctreeFile[:-4] + "_TENSOR" + activeOctreeFile[-4:], activeOut)

print("Export model file " + modelOctreeFile[:-4] + "_TENSOR" + modelOctreeFile[-4:] + " completed!" )
print("Export model file " + activeOctreeFile[:-4] + "_TENSOR" + activeOctreeFile[-4:] + " completed!" )

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
