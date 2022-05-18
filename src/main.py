# Libraries
import open3d as o3d
import numpy as np

# Classes
import DeformGraph as ed

if __name__ == '__main__':
    mesh = o3d.io.read_triangle_mesh("model/human_39.obj")
    # initialize a graph
    graph = ed.DGraph(mesh, sample_vox_size=0.1, max_dist_neigh=0.2, k_neigh=10)

    # Check number of each component of graph
    print('Vertices numbers: ' + str(len(graph.vertices)))
    print('Normals numbers: ' + str(len(graph.normals)))
    print('Faces numbers: ' + str(len(graph.faces)))

    # Show this graph
    graph.visualization(3)
