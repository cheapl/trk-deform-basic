# Libraries
import open3d as o3d
import numpy as np
import copy

from dgraph import DGraph
from icp import icp
import deformer
import utility

if __name__ == '__main__':
    ref_mesh = o3d.io.read_triangle_mesh("model/human_0.obj")
    #tar_mesh = o3d.io.read_triangle_mesh("model/human_0_rt.obj")
    tar_mesh = o3d.io.read_triangle_mesh("model/human_1.obj")
    # initialize a graph
    ref_graph = DGraph(ref_mesh, sample_vox_size=0.1, max_dist_neigh=0.2, k_nn=10, k_vn=4)

    # Check number of each component of graph
    print('Vertices numbers: ' + str(len(ref_graph.vertices)))
    print('Faces numbers: ' + str(len(ref_graph.faces)))

    affine_transform = icp(ref_graph, tar_mesh, np.eye(4))
    print(affine_transform)
    ref_graph.deform(0,affine_transform)

    error = utility.ref_to_tar_error(ref_graph, tar_mesh)
    print(error)
    utility.draw_registration_result(ref_graph, tar_mesh)

    # Test deformer
    #utility.visualization(ref_graph, 1)

