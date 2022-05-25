import open3d as o3d
import numpy as np
import copy

import utility

def icp(tef_graph, tar_mesh, trans_init=np.eye(4)):
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()

    source.points = o3d.utility.Vector3dVector(tef_graph.vertices)
    target.points = tar_mesh.vertices

    threshold = 0.05
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))

    return reg_p2p.transformation
