import open3d as o3d
import numpy as np
import copy
import math
import scipy.optimize

import utility
import deformer

class Optimizer:
    def __init__(self,ref_graph,tar_mesh):
        # global
        self.Nfeval = 1
        self.ref_graph = ref_graph
        self.tar_mesh = tar_mesh
        self.vertex_neighbours_coeffs = deformer.compute_vertex_neigh_coeffs(ref_graph)

    # Help function



# Objective function
def energy(xs,opt):
    vertices_num = len(opt.ref_graph.vertices)
    node_num = len(opt.ref_graph.node_positions)
    k_vn = opt.ref_graph.k_vn
    coeffs = opt.vertex_neighbours_coeffs
    source_pos = opt.ref_graph.vertices
    target_pos = np.asarray(opt.tar_mesh.vertices)
    #print(opt.ref_graph.vertex_neighbours_pos[[1,2,4]])

    #print(error)
    return 0

# Minimization
def non_rigid_align(x0,opt):
    p = scipy.optimize.minimize(fun=energy, x0=x0, args=(opt)).x
    return p