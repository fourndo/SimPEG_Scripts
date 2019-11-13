SimPEG_Scripts
==============

Repository of scripts to run SimPEG inversions and forwards

To run an inversion, open a command terminal and type:

``python PF_inversion.py your_project_directory\PF_inversion_input.json``

Mandatory
---------

* "data_type": Data file format, either "ubc_grav", "ubc_mag"
* "inversion_type": Inversion type, one of 'grav', 'mag', 'mvi', or 'mvis'
* "core_cell_size": [dx,dy,dz] Dimensions of the smallest cell size in the mesh.

Optional
--------

[DEFAULT]

* "result_folder": Directory used to output the results ["SimPEG_PFInversion"]
* "detrend": ["points", order] Remove trend from the data. 
			  "points" is either "all" (all points), or "corners"
			  order is integer 0, 1, or 2 for trend order
* "new_uncert": [Percent, floor] Values to be used for uncertainties.
* "show_graphics": [boolean], Show graphic plots [false]
* "no_data_value": [value], Value to use for no-data-value [-100]
* "parallelized": [boolean], Use dask parallelization [true]
* "max_chunk_size": [value], Size of data chunks to store in memory [128]
* "depth_core": ["auto", value] Compute mesh depth as value * survey width.
                [value] Set mesh core depth to value. [0]
* "model_reference": ["filename"], Load reference model from file.
                     [value], Reference property, scalar [0]
                     [value, value, value], Reference property, vector [0, 0, 0]
    				 If scalar input used for vector model, assume scalar amplitude in inducing field direction.
* "model_start": ["filename"], Load starting model from file.
                 [value], Start property, scalar [1e-4]. 
                 [value, value, value], Start property, vector [1e-4, 1e-4, 1e-4]
				 If scalar input used for vector model, assume scalar amplitude in inducing field direction.
* "alphas": [value, value, value, value], Alpha weights, can specify 4 or 12 values as required. [1,1,1,1,1,1,1,1,1,1,1,1]
* "target_chi": [value], target chi factor [1]
* "drape_data": [vlaue], smoothly drape the data over topography at altitude value


Magnetic only
--------

* "inducing_field_aid": [TOTAL FIELD, DIP, AZIMUTH], New inducing field as floats


More documentation to come in 2019!!
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
