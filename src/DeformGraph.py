import open3d as o3d
import numpy as np
import random

import math



class DGraph:
    def __init__(self, mesh, sample_vox_size, max_dist_neigh, k_nn, k_vn):
        # Vertices, vertex_normals and triangles in numpy forms for easy computation
        self.vertices = np.asarray(mesh.vertices)
        self.normals = np.asarray(mesh.vertex_normals)
        self.faces = np.asarray(mesh.triangles)
        # Sample vertices to get graph nodes
        self.node_positions, self.node_normals = self.mesh_vox_sampling(sample_vox_size)
        # Initialize rotation matrices and transformation matrices for nodes
        self.node_rots = np.ones((len(self.node_positions), 3, 3))
        self.node_trans = np.zeros((len(self.node_positions), 3))
        # Parameters for building neighbours
        self.max_dist_neigh = max_dist_neigh
        self.k_nn = k_nn
        self.k_vn = k_vn
        # Build neighbours for nodes
        self.node_neighbours = self.build_node_neigh()
        # Build neighbours for vertices
        self.vertex_neighbours = self.build_vertex_neigh()
        self.vertex_neigh_coeffs = self.compute_vertex_neigh_coeffs()

        print(self.vertex_neigh_coeffs)

    def mesh_vox_sampling(self, vox_size):
        # Construct point cloud use vertices of mesh
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.vertices)
        pcd.normals = o3d.utility.Vector3dVector(self.normals)
        # o3d.visualization.draw_geometries([pcd])
        # downsample pointcloud with a voxel
        pcd_ds, trace, idx = pcd.voxel_down_sample_and_trace(vox_size, pcd.get_min_bound(), pcd.get_max_bound(), False)
        print(pcd_ds)
        # Return nodes
        return np.asarray(pcd_ds.points), np.asarray(pcd_ds.normals)

    def build_node_neigh(self):
        # Initialize list of nodes' neighbours nodes
        node_neighbours = []
        # Generate pcd for nodes
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.node_positions)
        pcd.normals = o3d.utility.Vector3dVector(self.node_normals)
        #pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(self.node_positions), 3)))
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        # Do KNN search and radius search based on KDTrees
        for i in range(len(self.node_positions)):
            [k_ids, idx_dis, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], self.max_dist_neigh)
            [k_nn, idx_nn, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], self.k_nn)
            idx = list(set(idx_dis).intersection(set(idx_nn)))
            idx.remove(i)
            node_neighbours.append(idx)
        # Return result
        return node_neighbours

    def build_vertex_neigh(self):
        # Initialize list of vertices' neighbours nodes
        vertex_neighbours = []
        # Generate pcd for nodes
        node_pcd = o3d.geometry.PointCloud()
        node_pcd.points = o3d.utility.Vector3dVector(self.node_positions)
        node_pcd.normals = o3d.utility.Vector3dVector(self.node_normals)
        #node_pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(self.node_positions), 3)))
        pcd_tree = o3d.geometry.KDTreeFlann(node_pcd)
        # Generate pcd for vertice
        vertex_pcd = o3d.geometry.PointCloud()
        vertex_pcd.points = o3d.utility.Vector3dVector(self.vertices)
        vertex_pcd.normals = o3d.utility.Vector3dVector(self.normals)
        #vertex_pcd.colors = o3d.utility.Vector3dVector(np.ones((len(self.vertices), 3)))
        # Do KNN search  based on KDTrees
        for i in range(len(self.vertices)):
            [k, idx, _] = pcd_tree.search_knn_vector_3d(vertex_pcd.points[i], self.k_vn + 1)
            vertex_neighbours.append(list(idx))
        # Return result
        #print(vertex_neighbours)
        return vertex_neighbours

    def compute_vertex_neigh_coeffs(self):
        vertex_neigh_coeffs = []
        for i in range(len(self.vertices)):
            wj_vi = []
            neighbours = self.vertex_neighbours[i]
            vertex_pos = self.vertices[i]
            k_plus_1_neigh_pos = self.node_positions[neighbours[self.k_vn]]
            d_max = np.linalg.norm(vertex_pos-k_plus_1_neigh_pos)
            for j in range(self.k_vn):
                neigh_pos = self.node_positions[neighbours[j]]
                temp = 1 - ( np.linalg.norm(vertex_pos-neigh_pos) / d_max )
                w_j = math.pow(temp,2)
                wj_vi.append(w_j)
            vertex_neigh_coeffs.append(wj_vi)
        # Normalize
        np_coeffs = np.asarray(vertex_neigh_coeffs)
        sum_of_rows = np_coeffs.sum(axis=1)
        normalized_coeffs = np_coeffs / sum_of_rows[:, np.newaxis]
        return normalized_coeffs

    def visualization(self, mode):
        # initialize a new mesh for visualization
        rec_mesh = o3d.geometry.TriangleMesh()
        # Copy vertices, normals and faces of graph to this new mesh
        rec_mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        rec_mesh.vertex_normals = o3d.utility.Vector3dVector(self.normals)
        rec_mesh.triangles = o3d.utility.Vector3iVector(self.faces)
        # initialize a new pdc and visualization
        rec_pcd = o3d.geometry.PointCloud()
        rec_pcd.points = o3d.utility.Vector3dVector(self.node_positions)
        rec_pcd.normals = o3d.utility.Vector3dVector(self.node_normals)
        # Render it
        rec_mesh.compute_vertex_normals()
        if (mode == 0):
            o3d.visualization.draw_geometries([rec_mesh, rec_pcd])
        elif (mode == 1):
            o3d.visualization.draw_geometries([rec_mesh])
        elif (mode == 2):
            o3d.visualization.draw_geometries([rec_pcd])
        elif (mode == 3):
            # build line use node neighbours
            lines = []
            for i in range(len(self.node_neighbours)):
                for j in range(len(self.node_neighbours[i])):
                    if (self.node_neighbours[i][j] != i):
                        lines.append([i, self.node_neighbours[i][j]])
            lset = o3d.geometry.LineSet(o3d.utility.Vector3dVector(self.node_positions),
                                        o3d.utility.Vector2iVector(lines))
            o3d.visualization.draw_geometries([rec_pcd,lset])
