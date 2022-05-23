# Libraries
import open3d as o3d
import numpy as np
import math

# Classes
import DeformGraph as ed

def compute_vertex_neigh_coeffs(graph, method):
    # Method from original deformation node
    if method == 0:
        vertex_neigh_coeffs = []
        for i in range(len(graph.vertices)):
            wj_vi = []
            neighbours = graph.vertex_neighbours[i]
            vertex_pos = graph.vertices[i]
            k_plus_1_neigh_pos = graph.node_positions[neighbours[graph.k_vn]]
            d_max = np.linalg.norm(vertex_pos - k_plus_1_neigh_pos)
            for j in range(graph.k_vn):
                neigh_pos = graph.node_positions[neighbours[j]]
                temp = 1 - (np.linalg.norm(vertex_pos - neigh_pos) / d_max)
                w_j = math.pow(temp, 2)
                wj_vi.append(w_j)
            vertex_neigh_coeffs.append(wj_vi)
        # Normalize
        np_coeffs = np.asarray(vertex_neigh_coeffs)
        sum_of_rows = np_coeffs.sum(axis=1)
        normalized_coeffs = np_coeffs / sum_of_rows[:, np.newaxis]
        return normalized_coeffs
    # method from Fusion4D
    elif method == 1:
        vertex_neigh_coeffs = []
        for i in range(len(graph.vertices)):
            w_km = []
            neighbours = graph.vertex_neighbours[i]
            vertex_pos = graph.vertices[i]
            for j in range(graph.k_vn):
                neigh_pos = graph.node_positions[neighbours[j]]
                dist = np.linalg.norm(vertex_pos - neigh_pos)
                w_k = math.exp( -1 * (math.pow(dist,1) / (2 * math.pow(graph.theta,2))) )
                w_km.append(w_k)
            vertex_neigh_coeffs.append(w_km)
        # Normalize
        np_coeffs = np.asarray(vertex_neigh_coeffs)
        sum_of_rows = np_coeffs.sum(axis=1)
        normalized_coeffs = np_coeffs / sum_of_rows[:, np.newaxis]
        #print(normalized_coeffs)
        return normalized_coeffs
