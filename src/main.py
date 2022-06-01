# Libraries
import open3d as o3d
import numpy as np
import copy

from dgraph import DGraph
from optimizer import Optimizer,energy,non_rigid_align
from icp import icp
import deformer
import utility

if __name__ == '__main__':
    ref_mesh = o3d.io.read_triangle_mesh("model/human_0.obj")
    tar_mesh = o3d.io.read_triangle_mesh("model/human_1.obj")
    print(ref_mesh)
    # initialize a graph
    ref_graph = DGraph(ref_mesh, sample_vox_size=0.1, max_dist_neigh=0.2, k_nn=10, k_vn=4)

    # Rigid icp
    affine_transform = icp(ref_graph, tar_mesh, np.eye(4))
    ref_graph.deform(0,affine_transform)

    # Non-rigid alignment

    # TBD








