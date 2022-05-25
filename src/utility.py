import open3d as o3d
import numpy as np
import copy

def visualization(graph, mode):
    # initialize a new mesh for visualization
    rec_mesh = o3d.geometry.TriangleMesh()
    # Copy vertices, normals and faces of graph to this new mesh
    rec_mesh.vertices = o3d.utility.Vector3dVector(graph.vertices)
    rec_mesh.triangles = o3d.utility.Vector3iVector(graph.faces)
    # initialize a new pdc and visualization
    rec_pcd = o3d.geometry.PointCloud()
    rec_pcd.points = o3d.utility.Vector3dVector(graph.node_positions)
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
        for i in range(len(graph.node_neighbours)):
            for j in range(len(graph.node_neighbours[i])):
                if (graph.node_neighbours[i][j] != i):
                    lines.append([i, graph.node_neighbours[i][j]])
        lset = o3d.geometry.LineSet(o3d.utility.Vector3dVector(graph.node_positions),
                                    o3d.utility.Vector2iVector(lines))
        o3d.visualization.draw_geometries([rec_pcd, lset])

def graph_to_mesh(graph):
    rec_mesh = o3d.geometry.TriangleMesh()
    rec_mesh.vertices = o3d.utility.Vector3dVector(graph.vertices)
    rec_mesh.triangles = o3d.utility.Vector3iVector(graph.faces)
    return rec_mesh


def draw_registration_result(ref_graph, tar_mesh):
    ref_mesh = o3d.geometry.TriangleMesh()
    ref_mesh.vertices = o3d.utility.Vector3dVector(ref_graph.vertices)
    ref_mesh.triangles = o3d.utility.Vector3iVector(ref_graph.faces)

    tar_mesh = copy.deepcopy(tar_mesh)
    ref_mesh.paint_uniform_color([1, 0.706, 0])
    tar_mesh.paint_uniform_color([0, 0.651, 0.929])

    ref_mesh.compute_vertex_normals()
    tar_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([ref_mesh, tar_mesh])

def ref_to_tar_error(ref_graph,tar_mesh):
    rec_pcd = o3d.geometry.PointCloud()
    rec_pcd.points = o3d.utility.Vector3dVector(ref_graph.vertices)
    tar_pcd = o3d.geometry.PointCloud()
    tar_pcd.points = tar_mesh.vertices
    errors = rec_pcd.compute_point_cloud_distance(tar_pcd)
    #print(errors)
    np_errors = np.asarray(errors)
    error = np.sum(np_errors)
    #print(error)
    return error


def affine_to_rigidRT(graph, rigid_affine):
    node_num = len(graph.node_positions)
    print(rigid_affine)
    rot = rigid_affine[0:3, 0:3]
    tran = np.transpose(rigid_affine[0:3, 3:4])
    rots = np.broadcast_to(rot, (node_num, 3, 3))
    trans = np.broadcast_to(tran, (node_num, 3))
    return rots, trans

#def affine_to_deformRT(deform_affine):