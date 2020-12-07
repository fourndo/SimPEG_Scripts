# -*- coding: utf-8 -*-
"""
Created on Wed May  9 13:20:56 2018

@authors:
    fourndo@gmail.com
    orerocks@gmail.com


Potential field inversion
=========================

Run an inversion from input parameters stored in a json file.
See README for description of options


"""
import os
import sys
import json
import time
import multiprocessing
import dask
import numpy as np
import matplotlib.pyplot as plt
from discretize.utils import meshutils
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from scipy.spatial.ckdtree import cKDTree
import SimPEG.PF as PF
from SimPEG import (
    Mesh, Utils, Maps, Regularization,
    DataMisfit, Inversion, InvProblem, Directives, Optimization,
    )
from SimPEG.Utils import mkvc, matutils, modelutils

def decimate_survey_to_mesh(dec_mesh, data_trend, in_survey, full_survey=None):
    """Load a Geosoft grid and convert it to a Simpeg PF survey object

    Keyword arguments:
    dec_mesh -- A mesh to downsample against - usually a QuadTree based on the core data.
    data_trend -- the specific data_trend which will also need decimating
    in_survey -- The survey to be decimated. This may or may not be padded
    full_survey -- (optional). If only the data padding is to be decimated,
        then this contain the full padded survey and only points within this
        survey that are not in 'in_survey' will be decimated.

    """
    padded_survey_locs = in_survey.rxLoc[:, :2]
    if full_survey:
        core_survey_locs = full_survey.rxLoc[:, :2]

        # Identify which data points in the padded grid are actually padding
        data_tree = cKDTree(core_survey_locs)
        d, _ = data_tree.query(padded_survey_locs)
        # Find all padding points that are not in core_survey_locs:
        # they have distance > 0
        is_padding = d > 0.1
        is_padding_idx = np.where(is_padding)[0]

        print("  %.0f core data points (%.1f%%)" %
              (len(core_survey_locs),
               100 * len(core_survey_locs) / len(padded_survey_locs)))
        print("  %.0f padding points (%.1f%%)" %
              (sum(is_padding), 100 * sum(is_padding) / len(padded_survey_locs)))
    else:
        is_padding = np.full(in_survey.dobs.shape, True, dtype='bool')
        is_padding_idx = np.where(is_padding)[0]

    print("  %.0f total points" % len(padded_survey_locs))

    if full_survey and sum(is_padding) >= len(padded_survey_locs):

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(core_survey_locs[:, 0], core_survey_locs[:, 1],
                 'b+', markersize=1)
        ax1.plot(padded_survey_locs[:, 0], padded_survey_locs[:, 1],
                 'rx', markersize=1)
        ax1.set_aspect('equal')
        ax1.title.set_text('Max Entropy Padding')
        ax1.title.set_fontsize(20)
        ax1.legend(['Original', 'Padded'], loc=1, fontsize=20)

        ax2.plot(core_survey_locs[:, 0], core_survey_locs[:, 1],
                 'b+', markersize=10)
        ax2.plot(padded_survey_locs[:, 0], padded_survey_locs[:, 1],
                 'rx', markersize=10)
        ax2.set_aspect('equal')
        xr = np.percentile(core_survey_locs[:, 0], [0, 10])
        yr = np.percentile(core_survey_locs[:, 1], [0, 10])
        ax2.set_xlim(xr[0] - (np.diff(xr)), xr[1] + (np.diff(xr)))
        ax2.set_ylim(yr[0] - (np.diff(yr)), yr[1] + (np.diff(yr)))
        ax2.title.set_text('Max Entropy Padding (ZOOM)')
        ax2.title.set_fontsize(20)
        ax2.legend(['Original', 'Padded'], loc=1, fontsize=20)

        t = fig.get_size_inches()
        fig.set_size_inches(t[0]*5, t[1]*5)

        plt.show(block=False)

        assert(sum(is_padding) < len(padded_survey_locs)), \
        "Padding detection failed: Geosoft grid may have been shifted"

    # Get list of cell indices that contains each data point
    padding_data = in_survey.rxLoc[is_padding, :3]
    padding_obs = in_survey.dobs[is_padding]
    padding_std = in_survey.std[is_padding]
    # Get all cells that contain a padding data point
    c = dec_mesh.point2index(padding_data[:, :2])
    # Find unique cell indexes
    unique_cells = np.unique(c)
    # Get the center of each cell
    cell_centers = dec_mesh.gridCC[unique_cells]

    # Build a new data object to store the downsampled data
    rx_loc_new = np.nan*np.ones_like(in_survey.rxLoc)
    obs_new = np.nan*np.ones_like(in_survey.dobs)
    std_new = np.nan*np.ones_like(in_survey.std)
    # Track which data points are kept
    pad_points_kept = -1 * np.ones_like(in_survey.std, dtype='int64')
    count = 0
    for cell_ix, cell in enumerate(unique_cells):
        # Cycle through each cell that contains data
        # Get all data points within each cell
        data_in_current_cell_ix = (np.where(cell == c))[0]
        if len(data_in_current_cell_ix) > 1:
            # Find the data point closest to the cell center
            cell_tree = cKDTree(padding_data[data_in_current_cell_ix, :2])
            _, ii = cell_tree.query(cell_centers[cell_ix, :])
        else:
            # If only one data point, then use that!
            ii = 0

        # Get the coordinates of the point closest to cell center
        rx_loc_new[count, :] = padding_data[data_in_current_cell_ix[ii], :]
        # Calculate a new std based on average std and variability of data
        std_new[count] = np.nanmax([np.mean(padding_std[data_in_current_cell_ix]),
                                    np.std(padding_obs[data_in_current_cell_ix])])
        # And the most common dobs within the cell
        obs_new[count] = np.median(padding_obs[data_in_current_cell_ix])
        pad_points_kept[count] = is_padding_idx[data_in_current_cell_ix[ii]]
        count += 1

    # Create a new data object
    # By combining the decimated padding cells with
    # the original grid cell data (unchanged)
    mask = ~np.isnan(obs_new)
    rx_loc_new = np.vstack([rx_loc_new[mask, :], in_survey.rxLoc[~is_padding, :]])
    obs_new = np.r_[obs_new[mask], in_survey.dobs[~is_padding]]
    std_new = np.r_[std_new[mask], in_survey.std[~is_padding]]
    pad_points_kept2 = np.r_[pad_points_kept[mask],
                             np.arange(0, len(is_padding),
                                       dtype='int64')[~is_padding]]
    if isinstance(in_survey, PF.BaseMag.LinearSurvey):
        dec_obs = PF.BaseMag.RxObs(rx_loc_new)
        src_field = PF.BaseMag.SrcField([dec_obs], param=in_survey.srcField.param)
        in_survey_ds = PF.BaseMag.LinearSurvey(src_field, components=in_survey.components)
    else:
        dec_obs = PF.BaseGrav.RxObs(rx_loc_new)
        src_field = PF.BaseGrav.SrcField([dec_obs])
        in_survey_ds = PF.BaseGrav.LinearSurvey(src_field)

    in_survey_ds.dobs = obs_new
    in_survey_ds.std = std_new

    if not np.isscalar(data_trend):
        data_trend = data_trend[pad_points_kept2]

    print("  %.0f data points retained (%.1f%%)" %
          (len(in_survey_ds.dobs),
           100 * len(in_survey_ds.dobs) / len(padded_survey_locs)))

    return in_survey_ds, data_trend

def plot_convergence_curves(uncert, inversion_output, target_chi, out_dir, IRLS=None):
    """Plot inversion convergence curves, including IRLS information where available"""

    it = [d[0] for d in inversion_output]
    phi_d = [d[1]['phi_d'] for d in inversion_output]
    phi_m = [d[1]['phi_m'] for d in inversion_output]
    irlsiter = [d[1]['IRLSiterStart'] for d in inversion_output][-1]
    spherical = np.where([d[1]['coordinate_system'] in ['spherical'] for d in inversion_output])[0] + 1

    fig, axs = plt.figure(), plt.subplot()
    axs.plot(it, phi_d, 'ko-', lw=2)
    phi_d_target = 0.5*target_chi*len(uncert)
    left, right = plt.xlim()
    axs.plot(
        np.r_[left, right],
        np.r_[phi_d_target, phi_d_target], 'r--'
    )

    plt.yscale('log')

    bottom, top = plt.ylim()
    if len(spherical) > 0:
        axs.plot(
            np.r_[spherical[0], spherical[0]],
            np.r_[bottom, top], 'k--'
        )
        axs.text(
            spherical[0], top,
            'Spherical', va='top', ha='center',
            rotation='vertical', size=12,
            bbox={'facecolor': 'white'}
        )

    if irlsiter:
        axs.plot(
            np.r_[irlsiter, irlsiter],
            np.r_[bottom, top], 'k--'
        )
        axs.text(
            irlsiter, top,
            'IRLS', va='top', ha='center',
            rotation='vertical', size=12,
            bbox={'facecolor': 'white'}
        )

    twin = plt.twinx()
    twin.plot(it, phi_m, 'ko--', lw=2)
    plt.autoscale(enable=True, axis='both', tight=True)

    axs.set_ylabel(r'$\phi_d$', size=16, rotation=0)
    axs.set_xlabel('Iterations', size=14)
    twin.set_ylabel(r'$\phi_m$', size=16, rotation=0)
    axs.axis([left, right, bottom, top])
    plt.minorticks_on()
    t = fig.get_size_inches()
    fig.set_size_inches(t[0]*2, t[1]*2)
    fig.savefig(out_dir + 'Convergence_curve.png',
                bbox_inches='tight', dpi=300)
    plt.show(block=False)

dsep = os.path.sep
input_file = sys.argv[1]

if input_file is not None:
    workDir = dsep.join(
            os.path.dirname(os.path.abspath(input_file)).split(dsep)
            )
    if len(workDir) > 0:
        workDir += dsep
    else:
        workDir = os.getcwd() + dsep
    print('Working directory:', workDir)

else:
    assert input_file is not None, (
        "The input file is missing: 'python PFinversion.py input_file.json'"
    )

# Read json file and overwrite defaults
with open(input_file, 'r') as f:
    driver = json.load(f)

input_dict = {k.lower() if isinstance(k, str) else k:
              v.lower() if isinstance(v, str) else v for k, v in driver.items()}

assert "inversion_type" in list(input_dict.keys()), (
    "Require 'inversion_type' to be set: 'grav', 'mag', 'mvi', or 'mvis'"
)
assert input_dict["inversion_type"] in ['grav', 'mag', 'mvi', 'mvis'], (
    "'inversion_type' must be one of: 'grav', 'mag', 'mvi', or 'mvis'"
)

if "inversion_style" in list(input_dict.keys()):
    inversion_style = input_dict["inversion_style"]
else:
    inversion_style = "voxel"


if "result_folder" in list(input_dict.keys()):
    root = os.path.commonprefix([input_dict["result_folder"], workDir])
    outDir = workDir + os.path.relpath(input_dict["result_folder"], root) + dsep
else:
    outDir = workDir + dsep + "SimPEG_PFInversion" + dsep

os.system('mkdir ' + '"' + outDir + '"')
# extra quotes included in case path contains spaces

###############################################################################
# Deal with the data
if "inducing_field_aid" in list(input_dict.keys()):
    inducing_field_aid = np.asarray(input_dict["inducing_field_aid"])

    assert (len(inducing_field_aid) == 3 and inducing_field_aid[0] > 0), (
        "Inducing field must include H, INCL, DECL"
    )

else:
    inducing_field_aid = None

if "add_data_padding" in list(input_dict.keys()):
    add_data_padding = input_dict["add_data_padding"]

    assert (~add_data_padding), \
        "'add_data_padding' is currently only enabled for Geosoft Grid imports"
else:
    add_data_padding = False

if input_dict["data_type"] in ['ubc_grav']:

    survey = Utils.io_utils.readUBCgravityObservations(
        workDir + input_dict["data_file"]
    )

elif input_dict["data_type"] in ['ubc_mag']:

    survey, H0 = Utils.io_utils.readUBCmagneticsObservations(
        workDir + input_dict["data_file"]
    )
    survey.components = ['tmi']

elif input_dict["data_type"] in ['ftmg']:

    assert inducing_field_aid is not None, (
        "'inducing_field_aid' required for 'ftmg'"
    )
    patch = Utils.io_utils.readFTMGdataFile(
        file_path=workDir + input_dict["data_file"]
    )
    component = input_dict["ftmg_components"]
    H0 = np.asarray(input_dict["inducing_field_aid"])

    if "limits" in list(input_dict.keys()):
        limits = input_dict["limits"]
    else:
        limits = None

    xs, ys, zs = patch.getUtmLocations(limits=limits, ground=True)
    survey = patch.createFtmgSurvey(inducing_field=H0, force_comp=component)

    print(survey.components, component, survey.nD)
else:

    assert False, (
        "PF Inversion only implemented for 'data_type' of type:"
        " 'ubc_grav', 'ubc_mag', 'ftmg' "
    )

# Get data locations
rxLoc = survey.srcField.rxList[0].locs

# 0-level the data if required, data_trend = 0 level
if ("detrend" in list(input_dict.keys()) and
        input_dict["data_type"] in ['ubc_mag', 'ubc_grav']):

    for key, value in input_dict["detrend"].items():
        assert key in ["all", "corners"], "detrend key must be 'all' or 'corners'"
        assert value in [0, 1, 2], "detrend_order must be 0, 1, or 2"

        method = key
        order = value

    data_trend, _ = matutils.calculate_2D_trend(
        rxLoc, survey.dobs, order, method)

    survey.dobs -= data_trend

    if survey.std is None and "new_uncert" in list(input_dict.keys()):
        # In case uncertainty hasn't yet been set (e.g., geosoft grids)
        survey.std = np.ones(survey.dobs.shape)

    if input_dict["data_type"] in ['ubc_mag']:
        Utils.io_utils.writeUBCmagneticsObservations(
            os.path.splitext(
                    outDir + input_dict["data_file"])[0] + '_trend.mag',
                    survey, data_trend)
        Utils.io_utils.writeUBCmagneticsObservations(
            os.path.splitext(
                    outDir + input_dict["data_file"])[0] + '_detrend.mag',
                    survey, survey.dobs)
    else:
        Utils.io_utils.writeUBCgravityObservations(
            os.path.splitext(
                    outDir + input_dict["data_file"])[0] + '_trend.mag',
                    survey, data_trend)
        Utils.io_utils.writeUBCgravityObservations(
            os.path.splitext(
                    outDir + input_dict["data_file"])[0] + '_detrend.mag',
                    survey, survey.dobs)

else:
    data_trend = 0.0

# Update the specified data uncertainty
if (
    "new_uncert" in list(input_dict.keys())
):
    new_uncert = input_dict["new_uncert"]
    if new_uncert:
        assert (len(new_uncert) == 2 and all(np.asarray(new_uncert) >= 0)), (
                "New uncertainty requires pct fraction (0-1) and floor.")
        survey.std = np.maximum(abs(new_uncert[0]*survey.dobs), new_uncert[1])

if survey.std is None:
    survey.std = survey.dobs * 0 + 1  # Default


print('Min uncert: {0:.6g} nT'.format(survey.std.min()))

###############################################################################
# Manage other inputs
if "input_mesh_file" in list(input_dict.keys()):

    # Determine if the mesh is tensor or tree
    fid = open(workDir + input_dict["input_mesh_file"], 'r')
    for ii in range(6):
        line = fid.readline()
    fid.close()

    if line:
        input_mesh = Mesh.TreeMesh.readUBC(
            workDir + input_dict["input_mesh_file"]
        )
    else:
        input_mesh = Mesh.TensorMesh.readUBC(
            workDir + input_dict["input_mesh_file"]
        )

else:
    input_mesh = None

if "inversion_mesh_type" in list(input_dict.keys()):

    # Determine if the mesh is tensor or tree
    inversion_mesh_type = input_dict["inversion_mesh_type"]
else:

    if input_mesh is not None:
        inversion_mesh_type = input_mesh._meshType
    else:
        inversion_mesh_type = "TREE"

if "decimate_to_mesh" in list(input_dict.keys()):
    decimate_to_mesh = input_dict["decimate_to_mesh"]

    assert (inversion_mesh_type.upper() == "TREE"), \
        "'decimate_to_mesh' is currently only enabled for Octree meshes"

else:
    decimate_to_mesh = False

if "shift_mesh_z0" in list(input_dict.keys()):
    # Determine if the mesh is tensor or tree
    shift_mesh_z0 = input_dict["shift_mesh_z0"]
else:
    shift_mesh_z0 = None

if "topography" in list(input_dict.keys()):

    if isinstance(input_dict["topography"], str):
        topo = np.genfromtxt(workDir + input_dict["topography"],
                             skip_header=1)
    else:
        topo = survey.rxLoc.copy()
        topo[:, 2] = input_dict["topography"]

else:

    assert input_mesh is not None, (
        "You must either provide a 'topography' file of a 'input_mesh_file'"
        " in order to define an inversion volume"
    )
    # Grab the top coordinate and make a flat topo
    indTop = input_mesh.gridCC[:, 2] == input_mesh.vectorCCz[-1]
    topo = input_mesh.gridCC[indTop, :]
    topo[:, 2] += input_mesh.hz.min()/2. + 1e-8

# Create a linear interpolator for the mesh creation
topo_interp_function = NearestNDInterpolator(topo[:, :2], topo[:, 2])

if "drape_data" in list(input_dict.keys()):
    drape_data = input_dict["drape_data"]

    # In case topo is very large only use interpolant points next to obs
    max_pad_distance = [4 * drape_data]

    # Create new data locations draped at drapeAltitude above topo
    ix = (
        (topo[:, 0] >= (rxLoc[:, 0].min() - max_pad_distance)) &
        (topo[:, 0] <= (rxLoc[:, 0].max() + max_pad_distance)) &
        (topo[:, 1] >= (rxLoc[:, 1].min() - max_pad_distance)) &
        (topo[:, 1] <= (rxLoc[:, 1].max() + max_pad_distance))
    )

    F = LinearNDInterpolator(topo[ix, :2], topo[ix, 2])

    z = F(rxLoc[:, 0], rxLoc[:, 1]) + input_dict["drape_data"]
    survey.srcField.rxList[0].locs = np.c_[rxLoc[:, :2], z]
else:
    drape_data = None

if "target_chi" in list(input_dict.keys()):
    target_chi = input_dict["target_chi"]
else:
    target_chi = 1

if "model_norms" in list(input_dict.keys()):
    model_norms = input_dict["model_norms"]

else:
    model_norms = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

model_norms = np.c_[model_norms]

if input_dict["inversion_type"] in ['grav', 'mag']:
    model_norms = model_norms[:4]
    assert model_norms.shape[0] == 4, (
        "Model norms need at least for values (p_s, p_x, p_y, p_z)"
    )
else:
    assert model_norms.shape[0] == 12, (
        "Model norms needs 12 terms for [a, t, p] x [p_s, p_x, p_y, p_z]"
    )

if (
    "max_irls_iterations" in list(input_dict.keys()) and not
    input_dict["inversion_type"] == 'mvis'
):

    max_irls_iterations = input_dict["max_irls_iterations"]
    assert max_irls_iterations >= 0, "Max IRLS iterations must be >= 0"
else:
    if (input_dict["inversion_type"] == 'mvi') or (np.all(model_norms == 2)):
        # Cartesian or not sparse
        max_irls_iterations = 10

    else:
        # Spherical or sparse
        max_irls_iterations = 20

if "max_global_iterations" in list(input_dict.keys()):
    max_global_iterations = input_dict["max_global_iterations"]
    assert max_global_iterations >= 0, "Max IRLS iterations must be >= 0"
else:
    # Spherical or sparse
    max_global_iterations = 100

if "gradient_type" in list(input_dict.keys()):
    gradient_type = input_dict["gradient_type"]
else:
    gradient_type = 'total'

if "initial_beta" in list(input_dict.keys()):
    initial_beta = input_dict["initial_beta"]
else:
    initial_beta = None

if "n_cpu" in list(input_dict.keys()):
    n_cpu = input_dict["n_cpu"]
else:
    n_cpu = multiprocessing.cpu_count()

if "max_ram" in list(input_dict.keys()):
    max_ram = input_dict["max_ram"]
else:
    max_ram = 4

if "padding_distance" in list(input_dict.keys()):
    padding_distance = input_dict["padding_distance"]
else:
    padding_distance = [[0, 0], [0, 0], [0, 0]]

if "octree_levels_topo" in list(input_dict.keys()):
    octree_levels_topo = input_dict["octree_levels_topo"]
else:
    octree_levels_topo = [0, 1]

if "octree_levels_obs" in list(input_dict.keys()):
    octree_levels_obs = input_dict["octree_levels_obs"]
else:
    octree_levels_obs = [5, 5]

if "octree_levels_padding" in list(input_dict.keys()):
    octree_levels_padding = input_dict["octree_levels_padding"]
else:
    octree_levels_padding = [2, 2]

if "alphas" in list(input_dict.keys()):
    alphas = input_dict["alphas"]
    if len(alphas) == 4:
        alphas = alphas + [1, 1, 1, 1, 1, 1, 1, 1]
    else:
        assert len(alphas) == 12, "Alphas require list of 4 or 12 values"
else:
    alphas = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

if "alphas_mvis" in list(input_dict.keys()):
    alphas_mvis = input_dict["alphas_mvis"]
else:
    alphas_mvis = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]

if "input_vector_type" in list(input_dict.keys()):
    assert (
        input_dict["input_vector_type"] in ["xyz", "atp"]
        ), (
        "Input vector type needs to be xyz or atp"
    )
    input_vector_type = input_dict["input_vector_type"]
else:
    input_vector_type = "xyz"

if "model_start" in list(input_dict.keys()):

    if isinstance(input_dict["model_start"], str):
        model_start = input_dict["model_start"]

    else:
        model_start = np.r_[input_dict["model_start"]]
        assert model_start.shape[0] == 1 or model_start.shape[0] == 3, (
            "Start model needs to be a scalar or 3 component vector"
        )
else:
    model_start = [1e-4]

if "model_reference" in list(input_dict.keys()):

    if isinstance(input_dict["model_reference"], str):
        model_reference = input_dict["model_reference"]

    else:
        model_reference = np.r_[input_dict["model_reference"]]
        assert (
            model_reference.shape[0] == 1 or model_reference.shape[0] == 3
            ), (
            "Start model needs to be a scalar or 3 component vector"
        )
else:
    model_reference = [0.0]

if "lower_bound" in list(input_dict.keys()):
    lower_bound = input_dict["lower_bound"]
else:
    lower_bound = -np.inf

if "upper_bound" in list(input_dict.keys()):
    upper_bound = input_dict["upper_bound"]
else:
    upper_bound = np.inf

# @Nick: Not sure we want to keep this, not so transparent
if len(octree_levels_padding) < len(octree_levels_obs):
    octree_levels_padding += octree_levels_obs[len(octree_levels_padding):]

if "core_cell_size" in list(input_dict.keys()):
    core_cell_size = input_dict["core_cell_size"]
else:
    assert "'core_cell_size' must be added to the inputs"

if "depth_core" in list(input_dict.keys()):
    if "value" in list(input_dict["depth_core"].keys()):
        depth_core = input_dict["depth_core"]["value"]

    elif "auto" in list(input_dict["depth_core"].keys()):
        xLoc = survey.rxLoc[:, 0]
        yLoc = survey.rxLoc[:, 1]
        depth_core = (
            np.min([(xLoc.max()-xLoc.min()), (yLoc.max()-yLoc.min())]) *
            input_dict["depth_core"]["auto"]
        )
        print("Mesh core depth = %.2f" % depth_core)
    else:
        depth_core = 0
else:
    depth_core = 0

if "max_distance" in list(input_dict.keys()):
    max_distance = input_dict["max_distance"]
else:
    max_distance = np.inf

if "show_graphics" in list(input_dict.keys()):
    show_graphics = input_dict["show_graphics"]
else:
    show_graphics = False

if "max_chunk_size" in list(input_dict.keys()):
    max_chunk_size = input_dict["max_chunk_size"]
else:
    max_chunk_size = 128

if "chunk_by_rows" in list(input_dict.keys()):
    chunk_by_rows = input_dict["chunk_by_rows"]
else:
    chunk_by_rows = False

if "tiled_inversion" in list(input_dict.keys()):
    tiled_inversion = input_dict["tiled_inversion"]
else:
    tiled_inversion = True

if "tiling_method" in list(input_dict.keys()):
    tiling_method = 'orthogonal'
else:
    tiling_method = 'cluster'

if "output_tile_files" in list(input_dict.keys()):
    output_tile_files = input_dict["output_tile_files"]
else:
    output_tile_files = False

if input_dict["inversion_type"] in ['mvi', 'mvis', 'mvic']:
    vector_property = True
    n_blocks = 3
else:
    vector_property = False
    n_blocks = 1

if "no_data_value" in list(input_dict.keys()):
    no_data_value = input_dict["no_data_value"]
else:
    if vector_property:
        no_data_value = 0
    else:
        no_data_value = -100

if "parallelized" in list(input_dict.keys()):
    parallelized = input_dict["parallelized"]
else:
    parallelized = True

if parallelized:
    dask.config.set({'array.chunk-size': str(max_chunk_size) + 'MiB'})
    dask.config.set(scheduler='threads')
    dask.config.set(num_workers=n_cpu)

if add_data_padding or decimate_to_mesh:
    # Scipy.interpolate flags divide-by-zero and invalid value errors when making
    # the mesh. They don't affect the result, so we suppress them temporarily.
    old_settings = np.seterr(divide='ignore', invalid='ignore')
    # Create a quadtree mesh using the same params as the full mesh
    decimate_mesh = meshutils.mesh_builder_xyz(
        survey.rxLoc[:, :2],
        core_cell_size[:2],
        padding_distance=padding_distance[:2],
        mesh_type='tree'
    )

    decimate_mesh = meshutils.refine_tree_xyz(
        decimate_mesh, survey.rxLoc[:, :2], method='surface',
        max_distance=max_distance,
        octree_levels=octree_levels_obs,
        octree_levels_padding=octree_levels_padding,
        finalize=True,
    )
    np.seterr(**old_settings)

    if decimate_to_mesh:
        print("Decimating the whole survey to the mesh")
        survey, data_trend = decimate_survey_to_mesh(decimate_mesh,
                                                     data_trend, survey)
    elif add_data_padding:
        print("Decimating the padding points to the mesh")
        survey, data_trend = decimate_survey_to_mesh(decimate_mesh,
                                                     data_trend, survey, survey)

    Utils.io_utils.writeUBCmagneticsObservations(
    	outDir + os.path.splitext(
    	os.path.basename(input_dict["data_file"]))[0] + '_ds.mag',
    	survey, survey.dobs
    )

###############################################################################
# Processing

rxLoc = survey.rxLoc
# Create near obs topo
topo_elevations_at_data_locs = np.c_[
    rxLoc[:, :2],
    topo_interp_function(rxLoc[:, :2])
]

def createLocalMesh(rxLoc, ind_t):
    """
    Function to generate a mesh based on receiver locations
    """

    # Create new survey
    if input_dict["inversion_type"] == 'grav':
        rxLoc_t = PF.BaseGrav.RxObs(rxLoc[ind_t, :])
        srcField = PF.BaseGrav.SrcField([rxLoc_t])
        local_survey = PF.BaseGrav.LinearSurvey(srcField)
        local_survey.dobs = survey.dobs[ind_t]
        local_survey.std = survey.std[ind_t]
        local_survey.ind = ind_t

    elif input_dict["inversion_type"] in ['mag', 'mvi', 'mvis']:
        rxLoc_t = PF.BaseMag.RxObs(rxLoc[ind_t, :])
        srcField = PF.BaseMag.SrcField([rxLoc_t], param=survey.srcField.param)
        local_survey = PF.BaseMag.LinearSurvey(
            srcField, components=survey.components
        )

        dataInd = np.kron(ind_t, np.ones(len(survey.components))).astype('bool')

        local_survey.dobs = survey.dobs[dataInd]
        local_survey.std = survey.std[dataInd]
        local_survey.ind = ind_t

    meshLocal = meshutils.mesh_builder_xyz(
        topo_elevations_at_data_locs,
        core_cell_size,
        padding_distance=padding_distance,
        mesh_type=inversion_mesh_type,
        base_mesh=input_mesh,
        depth_core=depth_core
    )

    if shift_mesh_z0 is not None:
        print("In mesh z0")
        meshLocal.x0 = np.r_[meshLocal.x0[0], meshLocal.x0[1], shift_mesh_z0]

    if inversion_mesh_type.upper() == 'TREE':
        if topo is not None:
            meshLocal = meshutils.refine_tree_xyz(
                meshLocal, topo, method='surface',
                octree_levels=octree_levels_topo, finalize=False
            )

        meshLocal = meshutils.refine_tree_xyz(
            meshLocal, topo_elevations_at_data_locs[ind_t, :], method='surface',
            max_distance=max_distance,
            octree_levels=octree_levels_obs,
            octree_levels_padding=octree_levels_padding,
            finalize=True,
        )

    # Create combo misfit function
    return meshLocal, local_survey


if tiled_inversion:
    """
        LOOP THROUGH TILES

        Going through all problems:
        1- Pair the survey and problem
        2- Add up sensitivity weights
        3- Add to the global_misfit

        Create first mesh outside the parallel process

        Loop over different tile size and break problem until
        memory footprint false below max_ram
    """

    usedRAM = np.inf
    count = 1
    while usedRAM > max_ram:
        print("Tiling:" + str(count))

        # Default clustering algorithm goes slow on large data files,
        # so switch to simple method
        tiles, binCount, tileIDs, tile_numbers = \
            Utils.modelutils.tileSurveyPoints(rxLoc, count, method=tiling_method)


        # Grab the smallest bin and generate a temporary mesh
        indMax = np.argmax(binCount)

        X1, Y1 = tiles[0][:, 0], tiles[0][:, 1]
        X2, Y2 = tiles[1][:, 0], tiles[1][:, 1]

        ind_t = tileIDs == tile_numbers[indMax]

        # Create the mesh and refine the same as the global mesh
        meshLocal = meshutils.mesh_builder_xyz(
            topo_elevations_at_data_locs,
            core_cell_size,
            padding_distance=padding_distance,
            mesh_type=inversion_mesh_type,
            base_mesh=input_mesh,
            depth_core=depth_core
        )

        if shift_mesh_z0 is not None:
            meshLocal.x0 = np.r_[meshLocal.x0[0], meshLocal.x0[1], shift_mesh_z0]

        if inversion_mesh_type.upper() == 'TREE':
            if topo is not None:
                meshLocal = meshutils.refine_tree_xyz(
                    meshLocal, topo, method='surface',
                    octree_levels=octree_levels_topo, finalize=False
                )

            meshLocal = meshutils.refine_tree_xyz(
                meshLocal, topo_elevations_at_data_locs[ind_t, :],
                method='surface',
                max_distance=max_distance,
                octree_levels=octree_levels_obs,
                octree_levels_padding=octree_levels_padding,
                finalize=True,
            )

        tileLayer = Utils.surface2ind_topo(meshLocal, topo)

        # Calculate approximate problem size
        nDt, nCt = ind_t.sum()*1. * len(survey.components), tileLayer.sum()*1.

        nChunks = n_cpu  # Number of chunks
        cSa, cSb = int(nDt/nChunks), int(nCt/nChunks)  # Chunk sizes
        usedRAM = nDt * nCt * 8. * 1e-9
        count += 1
        print(nDt, nCt, usedRAM, binCount.min())
        del meshLocal

    nTiles = X1.shape[0]

    # Loop through the tiles and generate all sensitivities
    print("Number of tiles:" + str(nTiles))
    local_meshes, local_surveys = [], []
    for tt in range(nTiles):

        print("Tile " + str(tt+1) + " of " + str(X1.shape[0]))

        local_mesh, local_survey = createLocalMesh(
            rxLoc, tileIDs == tile_numbers[tt]
        )
        local_meshes += [local_mesh]
        local_surveys += [local_survey]

if (
    (input_mesh is None) or
    (input_mesh._meshType != inversion_mesh_type.upper())
):

    if tiled_inversion:
        print("Creating Global Octree")
        mesh = meshutils.mesh_builder_xyz(
            topo_elevations_at_data_locs, core_cell_size,
            padding_distance=padding_distance,
            mesh_type=inversion_mesh_type, base_mesh=input_mesh,
            depth_core=depth_core
            )

        if shift_mesh_z0 is not None:
            mesh.x0 = np.r_[mesh.x0[0], mesh.x0[1], shift_mesh_z0]

        if inversion_mesh_type.upper() == 'TREE':
            for local_mesh in local_meshes:

                mesh.insert_cells(
                    local_mesh.gridCC,
                    local_mesh.cell_levels_by_index(np.arange(local_mesh.nC)),
                    finalize=False
                )

            mesh.finalize()
    else:

        mesh, _ = createLocalMesh(rxLoc, np.ones(rxLoc.shape[0], dtype='bool'))

    # Transfer the refenrece and starting model to the new mesh

else:

    mesh = input_mesh

if show_graphics:
    # Plot a slice
    slicePosition = rxLoc[:, 1].mean()

    sliceInd = int(round(np.searchsorted(mesh.vectorCCy, slicePosition)))
    fig, ax1 = plt.figure(), plt.subplot()
    im = mesh.plotSlice(
        np.log10(mesh.vol),
        normal='Y', ax=ax1, ind=sliceInd,
        grid=True, pcolorOpts={"cmap": "plasma"}
    )
    ax1.set_aspect('equal')
    t = fig.get_size_inches()
    fig.set_size_inches(t[0]*2, t[1]*2)
    fig.savefig(
        outDir + 'Section_%.0f.png' % slicePosition,
        bbox_inches='tight', dpi=600
    )
    plt.show(block=False)

# Compute active cells
print("Calculating global active cells from topo")
activeCells = Utils.surface2ind_topo(mesh, topo)

if isinstance(mesh, Mesh.TreeMesh):
    Mesh.TreeMesh.writeUBC(
            mesh, outDir + 'OctreeMeshGlobal.msh',
            models={outDir + 'ActiveSurface.act': activeCells}
            )
else:
    mesh.writeModelUBC(
          'ActiveSurface.act', activeCells
    )

if "adjust_clearance" in list(input_dict.keys()):

    print("Forming cKDTree for clearance calculations")
    tree = cKDTree(mesh.gridCC[activeCells, :])

# Get the layer of cells directly below topo
nC = int(activeCells.sum())  # Number of active cells

# Create active map to go from reduce set to full
activeCellsMap = Maps.InjectActiveCells(
    mesh, activeCells, no_data_value, n_blocks=n_blocks
)


# Create reference and starting model
def get_model(input_value, vector=vector_property):
    # Loading a model file
    if isinstance(input_value, str):

        if not vector:
            model = input_mesh.readModelUBC(workDir + input_value)
            # model = np.c_[model]
        else:

            if "fld" in input_value:
                model = Utils.io_utils.readVectorUBC(mesh, workDir + input_value)
                if input_vector_type == 'atp':
                    # Flip the last vector back assuming ATP
                    model[:, -1] *= -1
                    model = Utils.matutils.atp2xyz(model)
            else:

                model = input_mesh.readModelUBC(workDir + input_value)
                model = Utils.sdiag(model) * np.kron(
                    Utils.matutils.dipazm_2_xyz(
                        dip=survey.srcField.param[1],
                        azm_N=survey.srcField.param[2]
                    ), np.ones((input_mesh.nC, 1))
                )
        # Transfer models from mesh to mesh
        if mesh != input_mesh:
            model = modelutils.transfer_to_mesh(input_mesh, mesh, model)
            print("Reference model transfered to new mesh!")

            file_name = outDir + input_value[:-4] + "_" + input_value[-4:]
            if not vector:
                if isinstance(mesh, Mesh.TreeMesh):
                    Mesh.TreeMesh.writeUBC(
                              mesh, outDir + 'OctreeMeshGlobal.msh',
                              models={file_name: model}
                            )
                else:
                    mesh.writeModelUBC(
                          file_name, model
                    )
            else:
                Utils.io_utils.writeVectorUBC(mesh, file_name, model)
    else:
        if not vector:
            model = np.ones(mesh.nC) * input_value[0]

        else:
            if np.r_[input_value].shape[0] == 3:
                # Assumes reference specified as: AMP, DIP, AZIM
                model = np.kron(np.c_[input_value], np.ones(mesh.nC)).T
                model = mkvc(
                    Utils.sdiag(model[:, 0]) *
                    Utils.matutils.dipazm_2_xyz(model[:, 1], model[:, 2])
                )
            else:
                # Assumes amplitude reference value in inducing field direction
                model = np.kron(
                    np.c_[
                        input_value[0], input_value[0],input_value[0]
                    ], np.ones(mesh.nC)
                )[0, :]

    return mkvc(model)

mref = get_model(model_reference)
mstart = get_model(model_start)

# Reduce to active set
if vector_property:
    mref = mref[np.kron(np.ones(3), activeCells).astype('bool')]
    mstart = mstart[np.kron(np.ones(3), activeCells).astype('bool')]
else:
    mref = mref[activeCells]
    mstart = mstart[activeCells]

# Homogeneous inversion only coded for scalar values for now
if (inversion_style == "homogeneous_units") and not vector_property:
    units = np.unique(mstart).tolist()

    # Build list of indecies for the geounits
    index = []
    for unit in units:
        index.append(mstart == unit)
    nC = len(index)

    # Collapse mstart and mref to the median reference values
    mstart = np.asarray([np.median(mref[mref == unit]) for unit in units])

    # Collapse mstart and mref to the median unit values
    mref = mstart.copy()

    model_map = Maps.SurjectUnits(index)
    regularization_map = Maps.IdentityMap(nP=nC)
    regularization_mesh = Mesh.TensorMesh([nC])
    regularization_actv = np.ones(nC, dtype='bool')
else:
    if vector_property:
        model_map = Maps.IdentityMap(nP=3*nC)
        regularization_map = Maps.Wires(('p', nC), ('s', nC), ('t', nC))
    else:
        model_map = Maps.IdentityMap(nP=nC)
        regularization_map = Maps.IdentityMap(nP=nC)
    regularization_mesh = mesh
    regularization_actv = activeCells

# Create identity map
if vector_property:
    global_weights = np.zeros(3*nC)
else:
    idenMap = Maps.IdentityMap(nP=nC)
    global_weights = np.zeros(nC)


def createLocalProb(meshLocal, local_survey, global_weights, ind):
    """
        CreateLocalProb(rxLoc, global_weights, lims, ind)

        Generate a problem, calculate/store sensitivities for
        given data points
    """

    # Need to find a way to compute sensitivities only for intersecting cells
    activeCells_t = np.ones(meshLocal.nC, dtype='bool')

    # Create reduced identity map
    if input_dict["inversion_type"] in ['mvi', 'mvis']:
        nBlock = 3
    else:
        nBlock = 1

    tile_map = Maps.Tile(
        (mesh, activeCells),
        (meshLocal, activeCells_t),
        nBlock=nBlock
    )

    activeCells_t = tile_map.activeLocal

    if "adjust_clearance" in list(input_dict.keys()):

        print("Setting Z values of data to respect clearance height")

        _, c_ind = tree.query(local_survey.rxLoc)
        dz = input_dict["adjust_clearance"]

        z = (
            mesh.gridCC[activeCells, 2][c_ind] +
            mesh.h_gridded[activeCells, 2][c_ind]/2 +
            dz
        )
        local_survey.srcField.rxList[0].locs[:, 2] = z

    if input_dict["inversion_type"] == 'grav':
        prob = PF.Gravity.GravityIntegral(
            meshLocal, rhoMap=tile_map*model_map, actInd=activeCells_t,
            parallelized=parallelized,
            Jpath=outDir + "Tile" + str(ind) + ".zarr",
            maxRAM=max_ram,
            n_cpu=n_cpu,
            max_chunk_size=max_chunk_size, chunk_by_rows=chunk_by_rows
            )

    elif input_dict["inversion_type"] == 'mag':
        prob = PF.Magnetics.MagneticIntegral(
            meshLocal, chiMap=tile_map*model_map, actInd=activeCells_t,
            parallelized=parallelized,
            Jpath=outDir + "Tile" + str(ind) + ".zarr",
            maxRAM=max_ram,
            n_cpu=n_cpu,
            max_chunk_size=max_chunk_size, chunk_by_rows=chunk_by_rows
            )

    elif input_dict["inversion_type"] in ['mvi', 'mvis']:
        prob = PF.Magnetics.MagneticIntegral(
            meshLocal, chiMap=tile_map*model_map, actInd=activeCells_t,
            parallelized=parallelized,
            Jpath=outDir + "Tile" + str(ind) + ".zarr",
            maxRAM=max_ram,
            modelType='vector',
            n_cpu=n_cpu,
            max_chunk_size=max_chunk_size, chunk_by_rows=chunk_by_rows
        )

    local_survey.pair(prob)
    # Data misfit function
    local_misfit = DataMisfit.l2_DataMisfit(local_survey)
    local_misfit.W = 1./local_survey.std

    wr = prob.getJtJdiag(np.ones_like(mstart), W=local_misfit.W)

    activeCellsTemp = Maps.InjectActiveCells(mesh, activeCells, 1e-8)

    global_weights += wr

    del meshLocal

    if output_tile_files:
        if input_dict["inversion_type"] == 'grav':

            Utils.io_utils.writeUBCgravityObservations(
                outDir + 'Survey_Tile' + str(ind) + '.dat',
                local_survey, local_survey.dobs
            )

        elif input_dict["inversion_type"] == 'mag':

            Utils.io_utils.writeUBCmagneticsObservations(
                outDir + 'Survey_Tile' + str(ind) + '.dat',
                local_survey, local_survey.dobs
            )

        Mesh.TreeMesh.writeUBC(
          mesh, outDir + 'Octree_Tile' + str(ind) + '.msh',
          models={outDir + 'JtJ_Tile' + str(ind) + ' .act': activeCellsTemp*wr[:nC]}
        )

    return local_misfit, global_weights


if tiled_inversion:
    for ind, (local_mesh, local_survey) in enumerate(zip(local_meshes, local_surveys)):

        print("Tile " + str(ind+1) + " of " + str(X1.shape[0]))

        local_misfit, global_weights = createLocalProb(local_mesh,
                                                       local_survey,
                                                       global_weights, ind)

        # Add the problems to a Combo Objective function
        if ind == 0:
            global_misfit = local_misfit

        else:
            global_misfit += local_misfit
else:

    global_misfit, global_weights = createLocalProb(mesh, survey, global_weights, 0)


# Global sensitivity weights (linear)
global_weights = global_weights**0.5
global_weights = (global_weights/np.max(global_weights))

if isinstance(mesh, Mesh.TreeMesh):
    Mesh.TreeMesh.writeUBC(
              mesh, outDir + 'OctreeMeshGlobal.msh',
              models={outDir + 'SensWeights.mod': \
                      (activeCellsMap*model_map*global_weights)[:mesh.nC]}
            )
else:
    mesh.writeModelUBC(
          'SensWeights.mod', (activeCellsMap*model_map*global_weights)[:mesh.nC]
    )

if not vector_property:

    # Create a regularization function
    reg = Regularization.Sparse(
        regularization_mesh,
        indActive=regularization_actv,
        mapping=regularization_map,
        alpha_s=alphas[0],
        alpha_x=alphas[1],
        alpha_y=alphas[2],
        alpha_z=alphas[3]
        )
    reg.norms = np.c_[model_norms].T
    reg.cell_weights = global_weights
    reg.mref = mref

else:

    # Create a regularization
    reg_p = Regularization.Sparse(
        mesh, indActive=activeCells,
        mapping=regularization_map.p,
        gradientType=gradient_type,
        alpha_s=alphas[0],
        alpha_x=alphas[1],
        alpha_y=alphas[2],
        alpha_z=alphas[3]
    )

    reg_p.cell_weights = (regularization_map.p * global_weights)
    reg_p.norms = np.c_[model_norms].T
    reg_p.mref = mref

    reg_s = Regularization.Sparse(
        mesh, indActive=activeCells,
        mapping=regularization_map.s,
        gradientType=gradient_type,
        alpha_s=alphas[4],
        alpha_x=alphas[5],
        alpha_y=alphas[6],
        alpha_z=alphas[7]
    )

    reg_s.cell_weights = (regularization_map.s * global_weights)
    reg_s.norms = np.c_[model_norms].T
    reg_s.mref = mref

    reg_t = Regularization.Sparse(
        mesh, indActive=activeCells,
        mapping=regularization_map.t,
        gradientType=gradient_type,
        alpha_s=alphas[8],
        alpha_x=alphas[9],
        alpha_y=alphas[10],
        alpha_z=alphas[11]
    )

    reg_t.cell_weights = (regularization_map.t * global_weights)
    reg_t.norms = np.c_[model_norms].T
    reg_t.mref = mref

    # Assemble the 3-component regularizations
    reg = reg_p + reg_s + reg_t

# Specify how the optimization will proceed, set susceptibility bounds to inf
opt = Optimization.ProjectedGNCG(
    maxIter=max_global_iterations,
    lower=lower_bound, upper=upper_bound,
    maxIterLS=20, maxIterCG=30, tolCG=1e-3,
    stepOffBoundsFact=1e-8,
    LSshorten=0.25
)

# Create the default L2 inverse problem from the above objects
invProb = InvProblem.BaseInvProblem(global_misfit, reg, opt, beta=initial_beta)

# Add a list of directives to the inversion
directiveList = []

if vector_property:
    directiveList.append(Directives.VectorInversion(
        inversion_type = input_dict["inversion_type"],
        chifact_target=1.)
    )

if initial_beta is None:
    directiveList.append(Directives.BetaEstimate_ByEig(beta0_ratio=1e+1))

# Pre-conditioner
directiveList.append(
    Directives.Update_IRLS(
        f_min_change=1e-4,
        maxIRLSiter=max_irls_iterations,
        minGNiter=1, beta_tol=0.5, prctile=90, floorEpsEnforced=True,
        coolingRate=1, coolEps_q=True, coolEpsFact=1.2,
        betaSearch=False
    )
)

directiveList.append(Directives.UpdatePreconditioner())

directiveList.append(
    Directives.SaveUBCModelEveryIteration(
        mapping=activeCellsMap * model_map,
        mesh=mesh,
        fileName=outDir + input_dict["inversion_type"],
        vector=input_dict["inversion_type"][0:3] == 'mvi'
    )
)

directiveList.append(
    Directives.SaveUBCPredictedEveryIteration(
        survey=survey,
        fileName=outDir + input_dict["inversion_type"],
        format=input_dict["inversion_type"]
    )
)
invProb_idx = len(directiveList) - 1

directiveList.append(
    Directives.SaveOutputDictEveryIteration()
)
inversion_output_idx = len(directiveList) - 1

# Put all the parts together
inv = Inversion.BaseInversion(
    invProb, directiveList=directiveList
)

# SimPEG reports half phi_d, so we scale to match
print(
    "Start Inversion: " + inversion_style +
    "\nTarget Misfit: %.2e (%.0f data with chifact = %g)" % (
        0.5 * target_chi * len(survey.std), len(survey.std), target_chi
    )
)

# Run the inversion
mrec = inv.run(mstart)

dpred = directiveList[invProb_idx].invProb.dpred

print("Target Misfit: %.3e (%.0f data with chifact = %g)" %
      (0.5*target_chi*len(survey.std), len(survey.std), target_chi))
print("Final Misfit:  %.3e" %
      (0.5 * np.sum(((survey.dobs - dpred)/survey.std)**2.)))

if show_graphics:
    plot_convergence_curves(survey.std, directiveList[inversion_output_idx].outDict.items(), target_chi, outDir)

if (len(np.shape(data_trend)) > 0) or (data_trend == 0):
    Utils.io_utils.writeUBCmagneticsObservations(
    outDir + 'Predicted_+trend.pre', survey, dpred+data_trend)

