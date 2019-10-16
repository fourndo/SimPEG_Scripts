SimPEG_Scripts
==============

Repository of scripts to run SimPEG inversions and forwards

To run an inversion, open a command terminal and type:

``python PF_inversion.py your_project_directory\PF_inversion_input.json``

More documentation to come in 2019!!
Stay tunes.


Notes on Octree Mesh
--------------------

.. image:: https://github.com/fourndo/SimPEG_Scripts/blob/master/Assets/Octree_refinement.png
    :alt: Mesh creation parameters

The following parameters can be used to modify the octree mesh.


 Parameters              |  Description
 ----------------------- | -----------------------------------------------------------------------
   core_cell_size 		  |  Smallest cell size dimension :math:`(h_x, h_y, h_z)`
   octree_levels_topo    | [*] Number of cells inserted below topography
   octree_levels_obs 	  | [*] Number of cells inserted below the data points
   octree_levels_padding | [*] Number of padding cells inserted horizontally around the data points
   max_distance  		  |  Maximum triangulation distance used by the refinement
   depth_core 		      |  Minimum depth of the mesh below the lowest point
   padding_distance 	  |  Minimum padding distance along the cartesian axes                                                    |


[*] List of integers ordered from the lowest octree level (smallest cell size)
to the highest. For instance the list [2, 6, 10] would require at least 2
fine cells (:math:`h_x`), followed by 6 cells at the :math:`2^{th}` level (:math:`2^1*h_x`) followed by
followed by 10 cells at the :math:`3^{th}` level (:math:`2^2*h_x`).
