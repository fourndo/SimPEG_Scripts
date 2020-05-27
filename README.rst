SimPEG_Scripts
==============

Repository of scripts to run SimPEG inversions and forwards

To run an inversion, open a command terminal and type:

``python PF_inversion.py your_project_directory\PF_inversion_input.json``

Parameters
----------

* ``data_type``: str
    Data file format,
     - ``ubc_grav``: See `GRAV3D documentation <https://grav3d.readthedocs.io/en/latest/content/files/obs.html#observations-file>`_
     - ``ubc_mag``: See `MAG3D documentation <https://mag3d.readthedocs.io/en/latest/content/files/obs.html#observations-file>`_
* ``inversion_type``: str
    Inversion type
        - 'grav': Invert for density in (g/cc).
        - 'mag': Invert for magnetic susceptibility in (SI).
        - 'mvi': Invert for effective susceptibility in Cartesian coordinates.
        - 'mvis': Invert for effective susceptibility in Spherical coordinates.
* ``core_cell_size``: list
    Dimensions of the smallest cell size in the mesh. [dx, dy, dz]

Optional settings: type = DEFAULT
---------------------------------

* ``add_data_padding``: bool = False
    Add an area of data padding around the input survey to manage edge effects
	*Currently only enabled for Geosoft grid imports*
* ``alphas``: list = [1, 1, 1, 1]
    Alpha weights used to scale the regularization functions.
        - Scalar (density, susceptibility): Requires 4 values for [a_s, a_x, a_y, a_z]
        - Vector (mvi): Requires 12 values for [a_s, a_x, a_y, a_z, t_s, t_x, t_y, t_z, p_s, p_x, p_y, p_z]
* ``alphas_mvis``: list = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
    Alpha weights used to scale the regularization functions only used for mvis inversions.
        - Vector (mvi): Requires 12 values for [a_s, a_x, a_y, a_z, t_s, t_x, t_y, t_z, p_s, p_x, p_y, p_z]
* ``chunk_by_rows``: bool = False
    Alternate memory management mode that can be faster for very large sensitivity or forward calculations
* ``depth_core``: dict: {str, float} = {"method": value}
    Thickness of core region defined by ``method``:
        - ``value``: Set ``value`` in meters
        - ``auto``: Compute mesh depth as: ``value`` * survey width.
* ``decimate_to_mesh``: bool = False
    Downsample the input data to at most 1 observation above each grid cell. This significantly reduces the problem size when using padding, as well as when ``core_cell_size`` is larger than the data spacing.
	*Currently only enabled for OcTree meshes*
* ``detrend``: dict {str: int} = None
    Remove trend from the data {method: order}.
        ``method``
            ``all``: All points used
            ``corners``: Points on the convex hull
        ``order``
            Integer defining the order of polynomial: 0, 1 or 2
* ``drape_data``: float = None
    Value defining the drape height above topography. If ``drape_data`` is used, the Z elevation of receivers are over-written
* ``input_mesh_file``: str = None
    Mesh file in UBC format used to load the ``model_reference`` and ``model_start``. The same mesh is used for the inversion if ``inversion_mesh_type`` is same type.
* ``inversion_mesh_type``: str = "tree"
    Type of mesh to be used in the inversion. If type differs from the ``input_mesh_file``, then the input ``model_reference`` and ``model_start`` are transfered over using a NearestNeighbour interpolation.
* ``inversion_style``: str
    Inversion style chosen from:
        - ``voxel``: Standard voxel base inversion [DEFAULT]
        - ``homogeneous_units``: Invert for best-fitting value for each domain defined by the unique values found in ``model_start``.
* ``lower_bound``: float = -Inf
    Value to use for lower bound in each cell
* ``max_chunk_size``: float = 128
        Size of data chunks to store in memory
* ``max_RAM``: float = 4
        Maximum available RAM. If ``tiled_inversion`` is True, then the tile size will be defined to keep the problem smaller than ``max_RAM``
* ``model_norms``: list = [2, 2, 2, 2]
    Model norms to apply, in range of 0-2 where 2 is least-squares
        - Scalar (density, susceptibility): Requires 4 values for [Lp_s, Lp_x, Lp_y, Lp_z]
        - Vector (mvi): Requires 12 values for [Lp_s, Lp_x, Lp_y, Lp_z, t_s, t_x, t_y, t_z, p_s, p_x, p_y, p_z]
* ``model_start``: str or float or list[float] = 1e-4
    Starting model to be loaded with the ``input_mesh``
        - str = ``filename``: Load starting model from file. If ``inversion_mesh_type`` differs from the ``input_mesh_file``, the model values are interpolated to the new mesh.
        - float = value: Start property, scalar [1e-4].
        - list = [value, value, value]: Start property model values for vector model [1e-4, 1e-4, 1e-4]
                 If scalar input used for vector model, assume scalar amplitude in inducing field direction.
* ``model_reference``: str or float or list[float] = 0
    Reference model to be loaded with the ``input_mesh``
        - str = ``filename``: Load starting model from file. If ``inversion_mesh_type`` differs from the ``input_mesh_file``, the model values are interpolated to the new mesh.
        - float = value: Start property, scalar [1e-4].
        - list = [value, value, value]: Start property model values for vector model [1e-4, 1e-4, 1e-4]
                 If scalar input used for vector model, assume scalar amplitude in inducing field direction.
* ``new_uncert``: list = [0, 1]
    List of values to be used for uncertainties set as [%, floor] (% from 0-1) where
    uncertainty = max(% * |data|, floor)
* ``no_data_value``: float = -100
    Value to use for no-data-value
* ``parallelized``: bool = True,
    Use dask parallelization
* ``result_folder``: str = "SimPEG_PFInversion"
    Directory used to output the results
* ``show_graphics``: bool = False
    Show graphic plots
* ``target_chi``: float = 1
    Target chi factor
* ``tiled_inversion``: bool = True,
    Use tiles to speed up the inversion and keep the problem small
* ``tiling_method``: str = 'cluster',
    Choice of methods to brake up the survey into tiles:
        - ``orthogonal``: Splits the survey by adding tiles of equal spatial dimensions.
        - ``clustering``: Use scikit-learn.AgglomerativeClustering algorithm (slow for large problems)
* ``upper_bound``: float = Inf
    Value to use for upper bound in each cell



Magnetic only
--------

* "inducing_field_aid": [TOTAL FIELD, DIP, AZIMUTH], New inducing field as floats


More documentation to come in 2020!!
Stay tunes.




Notes on Octree Mesh
--------------------

.. image:: https://github.com/fourndo/SimPEG_Scripts/blob/master/Assets/Octree_refinement.png
    :alt: Mesh creation parameters

The following parameters can be used to modify the octree mesh.


* **core_cell_size** :  Smallest cell size dimension :math:`(h_x, h_y, h_z)`
* **octree_levels_topo** : Number of cells inserted below topography [*]
* **octree_levels_obs** : Number of cells inserted below the data points [*]
* **octree_levels_padding** : Number of padding cells inserted horizontally around the data points [*]
* **max_distance** :  Maximum triangulation distance used by the refinement
* **depth_core** :  Minimum depth of the mesh below the lowest point
* **padding_distance** :  Minimum padding distance along the cartesian axes


[*] List of integers :math:`[nC_1, nC_2, ... ]` ordered from the lowest octree level (smallest cell size)
to the highest. For instance the list :math:`[2, 6, 10]` will request for at least 2
fine cells (:math:`h_x`), followed by 6 cells at the :math:`2^{th}` level (:math:`2^1*h_x`) followed by
followed by 10 cells at the :math:`3^{th}` level (:math:`2^2*h_x`).

See the `refine_tree_xyz <http://discretize.simpeg.xyz/en/master/api/generated/discretize.utils.refine_tree_xyz.html?highlight=refine#discretize-utils-refine-tree-xyz>`_ documentation for more details and examples.
