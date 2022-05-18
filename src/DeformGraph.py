import open3d as o3d
import numpy as np
import random


class DGraph:
    def __init__(self, mesh, sample_vox_size, max_dist_neigh, k_neigh):
        # Vertices, vertex_normals and triangles in numpy forms for easy computation
        self.vertices = np.asarray(mesh.vertices)
        self.normals = np.asarray(mesh.vertex_normals)
        self.faces = np.asarray(mesh.triangles)
        # Sample vertices to get graph nodes
        self.node_index, self.node_positions, self.node_normals = self.mesh_sampling(sample_vox_size)
        # Initialize rotation matrices and transformation matrices for nodes
        self.node_rots = np.ones((len(self.node_index), 3, 3))
        self.node_trans = np.zeros((len(self.node_index), 3))
        # Parameters for building neighbours
        self.max_dist_neigh = max_dist_neigh
        self.k_neigh = k_neigh
        # Build neighbours for nodes
        self.node_neighbours = self.build_neigh()

    def mesh_sampling(self, vox_size):
        # Construct point cloud use vertices of mesh
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.vertices)
        pcd.normals = o3d.utility.Vector3dVector(self.normals)
        # o3d.visualization.draw_geometries([pcd])
        # downsample pointcloud with a voxel
        pcd_ds, trace, idx = pcd.voxel_down_sample_and_trace(vox_size, pcd.get_min_bound(), pcd.get_max_bound(), False)
        print(pcd_ds)
        # print(idx)
        # determine index of sampled point in original vertices
        idx_list = []
        for int_vector in idx:
            idx_list.append(random.choice(int_vector))
        idx_list.sort()
        # visualize the sampled pcd
        points_ds = []
        normals_ds = []
        for idx in idx_list:
            points_ds.append(self.vertices[idx])
            normals_ds.append(self.normals[idx])
        # Return index of nodes
        return idx_list, np.asarray(points_ds), np.asarray(normals_ds)

    def build_neigh(self):
        # Initialize list of neighbours
        node_neighbours = []
        # Generate pcd for nodes
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.node_positions)
        pcd.normals = o3d.utility.Vector3dVector(self.node_normals)
        pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(self.node_index), 3)))
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        # Do KNN search and radius search based on KDTrees
        for i in range(len(self.node_index)):
            [k_ids, idx_dis, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], self.max_dist_neigh)
            [k_nn, idx_nn, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], self.k_neigh)
            idx = list(set(idx_dis).intersection(set(idx_nn)))
            node_neighbours.append(idx)
        # Return result
        return node_neighbours

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
        # build line use neighbours
        lines = []
        for i in range(len(self.node_neighbours)):
            for j in range(len(self.node_neighbours[i])):
                if (self.node_neighbours[i][j] != i):
                    lines.append([i, self.node_neighbours[i][j]])
        lset = o3d.geometry.LineSet(o3d.utility.Vector3dVector(self.node_positions), o3d.utility.Vector2iVector(lines))
        # Render it
        rec_mesh.compute_vertex_normals()
        if (mode == 0):
            o3d.visualization.draw_geometries([rec_mesh, rec_pcd])
        elif (mode == 1):
            o3d.visualization.draw_geometries([rec_mesh])
        elif (mode == 2):
            o3d.visualization.draw_geometries([rec_pcd])
        elif (mode == 3):
            o3d.visualization.draw_geometries([rec_pcd,lset])
