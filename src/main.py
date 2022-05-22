# Libraries
import open3d as o3d
import numpy as np

# Classes
import DeformGraph as ed
from icp import icp,draw_registration_result
from icp import icp,draw_registration_result

if __name__ == '__main__':
    ref_mesh = o3d.io.read_triangle_mesh("model/human_0.obj")
    tar_mesh = o3d.io.read_triangle_mesh("model/human_1.obj")
    # initialize a graph
    ref_graph = ed.DGraph(ref_mesh, sample_vox_size=0.1, max_dist_neigh=0.2, k_nn=10, k_vn=4)

    # Check number of each component of graph
    print('Vertices numbers: ' + str(len(ref_graph.vertices)))
    print('Normals numbers: ' + str(len(ref_graph.normals)))
    print('Faces numbers: ' + str(len(ref_graph.faces)))

    # Show this graph
    ref_graph.visualization(3)

    #initial_guess = np.eye(4)
    #affine_transform = icp(ref_mesh, tar_mesh, initial_guess)
    #print(affine_transform)
    #draw_registration_result(ref_mesh, tar_mesh, affine_transform)
