#!/usr/bin/env python3

from .data_types import FrontierGlobal, Pose2D


class FrontierFusion(object):
    def __init__(self, pose_registration, clustering_radius=1.0):
        self.pose_registration = pose_registration
        self.clustering_radius = float(clustering_radius)

    def fuse(self, robot_states):
        projected = []
        for state in robot_states:
            for frontier in state.local_frontiers:
                centroid_global = self.pose_registration.local_to_global(state.robot_id, frontier.centroid)
                if centroid_global is None:
                    continue
                projected.append(
                    FrontierGlobal(
                        frontier_id=frontier.frontier_id,
                        centroid=centroid_global,
                        size=frontier.size,
                        gain=frontier.gain,
                        source_robot_ids=set([state.robot_id]),
                        member_local_ids=[frontier.frontier_id],
                    )
                )
        return self._cluster_global_frontiers(projected)

    def _cluster_global_frontiers(self, projected):
        if not projected:
            return []

        clusters = []
        for frontier in projected:
            matched_cluster = None
            for cluster in clusters:
                if frontier.centroid.distance_xy(cluster["reference"]) <= self.clustering_radius:
                    matched_cluster = cluster
                    break

            if matched_cluster is None:
                clusters.append({"reference": frontier.centroid.copy(), "items": [frontier]})
            else:
                matched_cluster["items"].append(frontier)
                matched_cluster["reference"] = self._merge_centroid(
                    matched_cluster["items"],
                    sum(item.size for item in matched_cluster["items"]),
                )

        fused = []
        for index, cluster in enumerate(clusters):
            total_size = sum(item.size for item in cluster["items"])
            total_gain = sum(item.gain for item in cluster["items"])
            centroid = self._merge_centroid(cluster["items"], total_size)
            fused.append(
                FrontierGlobal(
                    frontier_id="global_frontier_{:03d}".format(index),
                    centroid=centroid,
                    size=total_size,
                    gain=total_gain,
                    source_robot_ids=set().union(*[item.source_robot_ids for item in cluster["items"]]),
                    member_local_ids=[member for item in cluster["items"] for member in item.member_local_ids],
                )
            )

        return fused

    def _merge_centroid(self, items, total_size):
        if total_size <= 0:
            return items[0].centroid.copy()
        return Pose2D(
            x=sum(item.centroid.x * item.size for item in items) / float(total_size),
            y=sum(item.centroid.y * item.size for item in items) / float(total_size),
            yaw=0.0,
        )
