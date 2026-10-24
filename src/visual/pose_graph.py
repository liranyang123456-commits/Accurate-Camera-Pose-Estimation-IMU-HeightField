"""
Backend Pose-Graph Optimization for drift mitigation across cumulative tracks.
(Section 3.2, Lemma 1, and Table 7 in manuscript)
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import open3d as o3d


class VisualPoseGraph:
    """
    Pose-graph backend using Open3D global optimization.
    """
    def __init__(self, max_correspondence_distance: float = 10.0):
        self.pose_graph = o3d.pipelines.registration.PoseGraph()
        self.max_corr_dist = max_correspondence_distance
        self.nodes_count = 0

    def add_node(self, pose_c2w: np.ndarray) -> int:
        """Add a pose graph node with initial 4x4 estimate."""
        node = o3d.pipelines.registration.PoseGraphNode(pose_c2w)
        self.pose_graph.nodes.append(node)
        node_id = self.nodes_count
        self.nodes_count += 1
        return node_id

    def add_edge(
        self,
        from_node: int,
        to_node: int,
        transform_from_to: np.ndarray,
        uncertainty: float = 1.0,
        is_loop: bool = False
    ) -> None:
        """
        Add a relative constraint edge between two nodes.
        """
        info_matrix = np.eye(6, dtype=np.float64) * (1.0 / max(uncertainty, 1e-4))
        edge = o3d.pipelines.registration.PoseGraphEdge(
            from_node,
            to_node,
            transform_from_to,
            info_matrix,
            uncertain=is_loop
        )
        self.pose_graph.edges.append(edge)

    def optimize(self, preference_loop_closure: float = 0.1) -> List[np.ndarray]:
        """
        Run Levenberg-Marquardt pose graph optimization and return refined node poses.
        """
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.max_corr_dist,
            edge_prune_threshold=0.25,
            preference_loop_weight=preference_loop_closure,
            reference_node=0
        )
        criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
        method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()

        o3d.pipelines.registration.global_optimization(
            self.pose_graph,
            method,
            criteria,
            option
        )

        optimized_poses = [np.asarray(node.pose) for node in self.pose_graph.nodes]
        return optimized_poses
