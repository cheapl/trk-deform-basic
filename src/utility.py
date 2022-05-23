import open3d as o3d

def visualization(graph, mode):
    # initialize a new mesh for visualization
    rec_mesh = o3d.geometry.TriangleMesh()
    # Copy vertices, normals and faces of graph to this new mesh
    rec_mesh.vertices = o3d.utility.Vector3dVector(graph.vertices)
    rec_mesh.vertex_normals = o3d.utility.Vector3dVector(graph.normals)
    rec_mesh.triangles = o3d.utility.Vector3iVector(graph.faces)
    # initialize a new pdc and visualization
    rec_pcd = o3d.geometry.PointCloud()
    rec_pcd.points = o3d.utility.Vector3dVector(graph.node_positions)
    rec_pcd.normals = o3d.utility.Vector3dVector(graph.node_normals)
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