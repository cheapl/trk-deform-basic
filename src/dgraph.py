import open3d as o3d
import numpy as np

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
        # Compute effective radius
        self.theta = self.compute_effec_r()

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

    def compute_effec_r(self):
        neigh_count = 0
        neigh_dist_sum = 0
        # Compute effective radius of the nodes(theta)
        for i in range(len(self.node_neighbours)):
            neighbours = self.node_neighbours[i]
            center_pos = self.node_positions[i]
            for neighbour in neighbours:
                neigh_count += 1
                neighbour_pos = self.node_positions[neighbour]
                neigh_dist = np.linalg.norm(center_pos - neighbour_pos)
                neigh_dist_sum += neigh_dist
        # d in skinning weights (Fusion4D 5.1)
        d = neigh_dist_sum / neigh_count
        #theta = 0.5 * d
        theta = d
        #print(theta)
        return theta
