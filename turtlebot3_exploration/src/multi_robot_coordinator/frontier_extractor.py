#!/usr/bin/env python3

import math
from collections import deque

from tf.transformations import euler_from_quaternion

from .data_types import FrontierLocal, Pose2D
from .tf_utils import compose_pose2d


class FrontierExtractor(object):
    def __init__(self, clustering_radius=0.6, min_cluster_size=5, occupied_threshold=50):
        self.clustering_radius = float(clustering_radius)
        self.min_cluster_size = int(min_cluster_size)
        self.occupied_threshold = int(occupied_threshold)

    def extract_frontiers(self, robot_id, map_msg):
        frontier_cells = self._find_frontier_cells(map_msg)
        clusters = self._cluster_cells(frontier_cells, map_msg.info.resolution)

        frontiers = []
        for index, cluster in enumerate(clusters):
            if len(cluster) < self.min_cluster_size:
                continue

            centroid, cells_world = self._cluster_centroid(map_msg, cluster)
            gain = self._estimate_gain(len(cluster), map_msg.info.resolution)
            frontier_id = "{}_frontier_{:03d}".format(robot_id, index)
            frontiers.append(
                FrontierLocal(
                    robot_id=robot_id,
                    frontier_id=frontier_id,
                    centroid=centroid,
                    size=len(cluster),
                    gain=gain,
                    cells=cells_world,
                )
            )

        return frontiers

    def _find_frontier_cells(self, map_msg):
        width = map_msg.info.width
        height = map_msg.info.height
        data = map_msg.data
        frontier_cells = set()

        for iy in range(1, height - 1):
            row_offset = iy * width
            for ix in range(1, width - 1):
                index = row_offset + ix
                if data[index] != 0:
                    continue
                if self._has_unknown_neighbor(ix, iy, width, height, data):
                    frontier_cells.add((ix, iy))

        return frontier_cells

    def _has_unknown_neighbor(self, ix, iy, width, height, data):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = ix + dx
                ny = iy + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if data[ny * width + nx] < 0:
                    return True
        return False

    def _cluster_cells(self, frontier_cells, resolution):
        if not frontier_cells:
            return []

        visited = set()
        clusters = []
        frontier_cells = set(frontier_cells)
        radius_in_cells = max(1, int(math.ceil(self.clustering_radius / max(resolution, 1e-6))))
        offsets = []
        for dy in range(-radius_in_cells, radius_in_cells + 1):
            for dx in range(-radius_in_cells, radius_in_cells + 1):
                if dx == 0 and dy == 0:
                    continue
                if math.hypot(dx, dy) <= radius_in_cells:
                    offsets.append((dx, dy))

        for cell in frontier_cells:
            if cell in visited:
                continue
            queue = deque([cell])
            cluster = []
            visited.add(cell)

            while queue:
                current = queue.popleft()
                cluster.append(current)
                for dx, dy in offsets:
                    neighbor = (current[0] + dx, current[1] + dy)
                    if neighbor in frontier_cells and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            clusters.append(cluster)

        return clusters

    def _cluster_centroid(self, map_msg, cluster):
        cells_world = [self._grid_to_world(map_msg, ix, iy) for ix, iy in cluster]
        centroid = Pose2D(
            x=sum(cell.x for cell in cells_world) / float(len(cells_world)),
            y=sum(cell.y for cell in cells_world) / float(len(cells_world)),
            yaw=0.0,
        )
        return centroid, cells_world

    def _estimate_gain(self, cluster_size, resolution):
        return float(cluster_size) * float(resolution)

    def _grid_to_world(self, map_msg, ix, iy):
        origin = map_msg.info.origin
        origin_yaw = euler_from_quaternion(
            [origin.orientation.x, origin.orientation.y, origin.orientation.z, origin.orientation.w]
        )[2]
        origin_pose = Pose2D(origin.position.x, origin.position.y, origin_yaw)
        offset_pose = Pose2D(
            x=(ix + 0.5) * map_msg.info.resolution,
            y=(iy + 0.5) * map_msg.info.resolution,
            yaw=0.0,
        )
        return compose_pose2d(origin_pose, offset_pose)
