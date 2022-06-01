# Libraries
import open3d as o3d
import numpy as np
import math
import copy

import  utility

def compute_vertex_neigh_coeffs(graph):
        vertex_neigh_coeffs = []
        for i in range(len(graph.vertices)):
            w_km = []
            neighbours = graph.vertex_neighbours[i]
            vertex_pos = graph.vertices[i]
            for j in range(graph.k_vn):
                neigh_pos = graph.node_positions[neighbours[j]]
                dist = np.linalg.norm(vertex_pos - neigh_pos)
                w_k = math.exp( -1 * (math.pow(dist,2) / (2 * math.pow(graph.effec_r/2,2))) )
                w_km.append(w_k)
            vertex_neigh_coeffs.append(w_km)
        # Normalize
        np_coeffs = np.asarray(vertex_neigh_coeffs)
        sum_of_rows = np_coeffs.sum(axis=1)
        normalized_coeffs = np_coeffs / sum_of_rows[:, np.newaxis]
        #print(normalized_coeffs)
        return normalized_coeffs

def deform_graph_non_rigid(graph):
    new_vertices = []
    coeffs = compute_vertex_neigh_coeffs(graph)
    for i in range(len(graph.vertices)):
        v_i = graph.vertices[i]
        neigh_nodes = graph.vertex_neighbours[i]
        coeff = coeffs[i]
        new_v_i = np.asarray([0.0,0.0,0.0])
        for j in range(len(neigh_nodes)):
            neigh_idx = neigh_nodes[j]
            g_j = graph.node_positions[neigh_idx]
            R_j = graph.node_rots[neigh_idx]
            t_j = graph.node_trans[neigh_idx]
            #temp = np.dot(R_j,np.transpose(v_i - g_j)) + g_j + t_j
            temp = np.dot(R_j, v_i - g_j) + g_j + t_j
            new_v_i  += coeff[j] * temp
        new_vertices.append(new_v_i)
    return np.asarray(new_vertices)

def deform_graph_rigid(graph,affine_transform):
    mesh = utility.graph_to_mesh(graph)
    mesh_t = copy.deepcopy(mesh).transform(affine_transform)
    return np.asarray(mesh_t.vertices)
