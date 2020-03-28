"""Adapted from rllab maze_env.py."""

import os
import tempfile
import xml.etree.ElementTree as ET
import math
import numpy as np
import gym
from . import maze_env_utils
from .point import PointEnv
from gym.utils import seeding

# Directory that contains mujoco xml files.
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'assets')


class PointMazeEnv(gym.Env):
    MODEL_CLASS = PointEnv

    MAZE_HEIGHT = None
    MAZE_SIZE_SCALING = None

    def __init__(
            self,
            maze_id=None,
            maze_height=0.5,
            maze_size_scaling=8,
            n_bins=0,
            sensor_range=3.,
            sensor_span=2 * math.pi,
            observe_blocks=False,
            put_spin_near_agent=False,
            top_down_view=False,
            manual_collision=False,
            goal=None,
            EPS=0.25,
            max_episode_steps=2000,
            *args,
            **kwargs):
        self._maze_id = maze_id

        model_cls = self.__class__.MODEL_CLASS
        if model_cls is None:
            raise "MODEL_CLASS unspecified!"
        xml_path = os.path.join(MODEL_DIR, model_cls.FILE)
        self.tree = tree = ET.parse(xml_path)
        self.worldbody = worldbody = tree.find(".//worldbody")
        self.visualize_goal = False
        self.max_episode_steps = max_episode_steps
        self.t = 0
        self.MAZE_HEIGHT = height = maze_height
        self.MAZE_SIZE_SCALING = size_scaling = maze_size_scaling
        self._n_bins = n_bins
        self._sensor_range = sensor_range * size_scaling
        self._sensor_span = sensor_span
        self._observe_blocks = observe_blocks
        self._put_spin_near_agent = put_spin_near_agent
        self._top_down_view = top_down_view
        self._manual_collision = manual_collision

        self.MAZE_STRUCTURE = structure = maze_env_utils.construct_maze(
            maze_id=self._maze_id)
        # Elevate the maze to allow for falling.
        self.elevated = any(-1 in row for row in structure)
        self.blocks = any(
            any(maze_env_utils.can_move(r) for r in row)
            for row in structure)  # Are there any movable blocks?

        torso_x, torso_y = self._find_robot()  # x, y coordinates
        self._init_torso_x = torso_x
        self._init_torso_y = torso_y
        self._init_positions = [
            (x - torso_x, y - torso_y)
            for x, y in self._find_all_robots()]

        self._view = np.zeros([5, 5, 3])

        height_offset = 0.
        if self.elevated:
            height_offset = height * size_scaling
            torso = tree.find(".//body[@name='torso']")
            torso.set('pos', '0 0 %.2f' % (0.75 + height_offset))
        if self.blocks:
            default = tree.find(".//default")
            default.find('.//geom').set('solimp', '.995 .995 .01')

        self.movable_blocks = []
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                struct = structure[i][j]
                if struct == 'r' and self._put_spin_near_agent:
                    struct = maze_env_utils.Move.SpinXY
                if self.elevated and struct not in [-1]:
                    # Create elevated platform.
                    ET.SubElement(
                        worldbody, "geom",
                        name="elevated_%d_%d" % (i, j),
                        pos="%f %f %f" % (j * size_scaling - torso_x,
                                          i * size_scaling - torso_y,
                                          height / 2 * size_scaling),
                        size="%f %f %f" % (0.5 * size_scaling,
                                           0.5 * size_scaling,
                                           height / 2 * size_scaling),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.9 0.9 0.9 1",
                    )
                if struct == 1:  # Unmovable block.
                    # Offset all coordinates so that robot starts at the origin
                    ET.SubElement(
                        worldbody, "geom",
                        name="block_%d_%d" % (i, j),
                        pos="%f %f %f" % (j * size_scaling - torso_x,
                                          i * size_scaling - torso_y,
                                          height_offset +
                                          height / 2 * size_scaling),
                        size="%f %f %f" % (0.5 * size_scaling,
                                           0.5 * size_scaling,
                                           height / 2 * size_scaling),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.4 0.4 0.4 1",
                    )
                elif maze_env_utils.can_move(struct):
                    name = "movable_%d_%d" % (i, j)
                    self.movable_blocks.append((name, struct))
                    falling = maze_env_utils.can_move_z(struct)
                    spinning = maze_env_utils.can_spin(struct)
                    x_offset = 0.25 * size_scaling if spinning else 0.0
                    y_offset = 0.0
                    shrink = 0.1 if spinning else 0.99 if falling else 1.0
                    height_shrink = 0.1 if spinning else 1.0
                    _x = j * size_scaling - torso_x + x_offset
                    _y = i * size_scaling - torso_y + y_offset
                    _z = height / 2 * size_scaling * height_shrink
                    movable_body = ET.SubElement(
                        worldbody, "body",
                        name=name,
                        pos="%f %f %f" % (_x, _y, height_offset + _z),
                    )
                    ET.SubElement(
                        movable_body, "geom",
                        name="block_%d_%d" % (i, j),
                        pos="0 0 0",
                        size="%f %f %f" % (0.5 * size_scaling * shrink,
                                           0.5 * size_scaling * shrink,
                                           _z),
                        type="box",
                        material="",
                        mass="0.001" if falling else "0.0002",
                        contype="1",
                        conaffinity="1",
                        rgba="0.9 0.1 0.1 1"
                    )
                    if maze_env_utils.can_move_x(struct):
                        ET.SubElement(
                            movable_body, "joint",
                            armature="0",
                            axis="1 0 0",
                            damping="0.0",
                            limited="true" if falling else "false",
                            range="%f %f" % (-size_scaling, size_scaling),
                            margin="0.01",
                            name="movable_x_%d_%d" % (i, j),
                            pos="0 0 0",
                            type="slide"
                        )
                    if maze_env_utils.can_move_y(struct):
                        ET.SubElement(
                            movable_body, "joint",
                            armature="0",
                            axis="0 1 0",
                            damping="0.0",
                            limited="true" if falling else "false",
                            range="%f %f" % (-size_scaling, size_scaling),
                            margin="0.01",
                            name="movable_y_%d_%d" % (i, j),
                            pos="0 0 0",
                            type="slide"
                        )
                    if maze_env_utils.can_move_z(struct):
                        ET.SubElement(
                            movable_body, "joint",
                            armature="0",
                            axis="0 0 1",
                            damping="0.0",
                            limited="true",
                            range="%f 0" % (-height_offset),
                            margin="0.01",
                            name="movable_z_%d_%d" % (i, j),
                            pos="0 0 0",
                            type="slide"
                        )
                    if maze_env_utils.can_spin(struct):
                        ET.SubElement(
                            movable_body, "joint",
                            armature="0",
                            axis="0 0 1",
                            damping="0.0",
                            limited="false",
                            name="spinable_%d_%d" % (i, j),
                            pos="0 0 0",
                            type="ball"
                        )

        torso = tree.find(".//body[@name='torso']")
        geoms = torso.findall(".//geom")
        for geom in geoms:
            if 'name' not in geom.attrib:
                raise Exception("Every geom of the torso must have a name "
                                "defined")

        _, file_path = tempfile.mkstemp(text=True, suffix='.xml')
        tree.write(file_path)

        self.wrapped_env = model_cls(*args, file_path=file_path, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.GOAL = goal
        if self.GOAL is not None:
            self.GOAL = self.unwrapped._rowcol_to_xy(*self.GOAL)
        self.EPS = EPS

    def get_ori(self):
        return self.wrapped_env.get_ori()

    def get_top_down_view(self):
        self._view = np.zeros_like(self._view)

        def valid(row, col):
            return self._view.shape[0] > row >= 0 \
                   and self._view.shape[1] > col >= 0

        def update_view(x, y, d, row=None, col=None):
            if row is None or col is None:
                x = x - self._robot_x
                y = y - self._robot_y

                row, col = self._xy_to_rowcol(x, y)
                update_view(x, y, d, row=row, col=col)
                return

            row, row_frac, col, col_frac = int(row), row % 1, int(col), col % 1
            if row_frac < 0:
                row_frac += 1
            if col_frac < 0:
                col_frac += 1

            if valid(row, col):
                self._view[row, col, d] += (
                        (min(1., row_frac + 0.5) - max(0., row_frac - 0.5)) *
                        (min(1., col_frac + 0.5) - max(0., col_frac - 0.5)))
            if valid(row - 1, col):
                self._view[row - 1, col, d] += (
                        (max(0., 0.5 - row_frac)) *
                        (min(1., col_frac + 0.5) - max(0., col_frac - 0.5)))
            if valid(row + 1, col):
                self._view[row + 1, col, d] += (
                        (max(0., row_frac - 0.5)) *
                        (min(1., col_frac + 0.5) - max(0., col_frac - 0.5)))
            if valid(row, col - 1):
                self._view[row, col - 1, d] += (
                        (min(1., row_frac + 0.5) - max(0., row_frac - 0.5)) *
                        (max(0., 0.5 - col_frac)))
            if valid(row, col + 1):
                self._view[row, col + 1, d] += (
                        (min(1., row_frac + 0.5) - max(0., row_frac - 0.5)) *
                        (max(0., col_frac - 0.5)))
            if valid(row - 1, col - 1):
                self._view[row - 1, col - 1, d] += (
                        (max(0., 0.5 - row_frac)) * max(0., 0.5 - col_frac))
            if valid(row - 1, col + 1):
                self._view[row - 1, col + 1, d] += (
                        (max(0., 0.5 - row_frac)) * max(0., col_frac - 0.5))
            if valid(row + 1, col + 1):
                self._view[row + 1, col + 1, d] += (
                        (max(0., row_frac - 0.5)) * max(0., col_frac - 0.5))
            if valid(row + 1, col - 1):
                self._view[row + 1, col - 1, d] += (
                        (max(0., row_frac - 0.5)) * max(0., 0.5 - col_frac))

        # Draw ant.
        robot_x, robot_y = self.wrapped_env.get_body_com("torso")[:2]
        self._robot_x = robot_x
        self._robot_y = robot_y
        self._robot_ori = self.get_ori()

        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING

        # Draw immovable blocks and chasms.
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 1:  # Wall.
                    update_view(j * size_scaling - self._init_torso_x,
                                i * size_scaling - self._init_torso_y,
                                0)
                if structure[i][j] == -1:  # Chasm.
                    update_view(j * size_scaling - self._init_torso_x,
                                i * size_scaling - self._init_torso_y,
                                1)

        # Draw movable blocks.
        for block_name, block_type in self.movable_blocks:
            block_x, block_y = self.wrapped_env.get_body_com(block_name)[:2]
            update_view(block_x, block_y, 2)

        import cv2
        cv2.imshow('x.jpg', cv2.resize(
            np.uint8(self._view * 255), (512, 512),
            interpolation=cv2.INTER_CUBIC))
        cv2.waitKey(0)

        return self._view

    def get_range_sensor_obs(self):
        """Returns egocentric range sensor observations of maze."""
        robot_x, robot_y, robot_z = self.wrapped_env.get_body_com("torso")[:3]
        ori = self.get_ori()

        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING
        height = self.MAZE_HEIGHT

        segments = []
        # Get line segments (corresponding to outer boundary) of each immovable
        # block or drop-off.
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] in [1, -1]:  # There's a wall or drop-off.
                    cx = j * size_scaling - self._init_torso_x
                    cy = i * size_scaling - self._init_torso_y
                    x1 = cx - 0.5 * size_scaling
                    x2 = cx + 0.5 * size_scaling
                    y1 = cy - 0.5 * size_scaling
                    y2 = cy + 0.5 * size_scaling
                    struct_segments = [
                        ((x1, y1), (x2, y1)),
                        ((x2, y1), (x2, y2)),
                        ((x2, y2), (x1, y2)),
                        ((x1, y2), (x1, y1)),
                    ]
                    for seg in struct_segments:
                        segments.append(dict(
                            segment=seg,
                            type=structure[i][j],
                        ))

        for block_name, block_type in self.movable_blocks:
            block_x, block_y, block_z = \
                self.wrapped_env.get_body_com(block_name)[:3]
            if (block_z + height * size_scaling / 2 >= robot_z and
                    robot_z >= block_z - height * size_scaling / 2):
                # Block in view.
                x1 = block_x - 0.5 * size_scaling
                x2 = block_x + 0.5 * size_scaling
                y1 = block_y - 0.5 * size_scaling
                y2 = block_y + 0.5 * size_scaling
                struct_segments = [
                    ((x1, y1), (x2, y1)),
                    ((x2, y1), (x2, y2)),
                    ((x2, y2), (x1, y2)),
                    ((x1, y2), (x1, y1)),
                ]
                for seg in struct_segments:
                    segments.append(dict(
                        segment=seg,
                        type=block_type,
                    ))

        # 3 for wall, drop-off, block
        sensor_readings = np.zeros((self._n_bins, 3))
        for ray_idx in range(self._n_bins):
            ray_ori = (ori - self._sensor_span * 0.5 + (
                    2 * ray_idx + 1.0) /
                       (2 * self._n_bins) * self._sensor_span)
            ray_segments = []
            # Get all segments that intersect with ray.
            for seg in segments:
                p = maze_env_utils.ray_segment_intersect(
                    ray=((robot_x, robot_y), ray_ori),
                    segment=seg["segment"])
                if p is not None:
                    ray_segments.append(dict(
                        segment=seg["segment"],
                        type=seg["type"],
                        ray_ori=ray_ori,
                        distance=maze_env_utils.point_distance(
                            p, (robot_x, robot_y)),
                    ))
            if len(ray_segments) > 0:
                # Find out which segment is intersected first.
                first_seg = sorted(
                    ray_segments, key=lambda x: x["distance"])[0]
                seg_type = first_seg["type"]
                idx = (0 if seg_type == 1 else  # Wall.
                       1 if seg_type == -1 else  # Drop-off.
                       2 if maze_env_utils.can_move(seg_type) else  # Block.
                       None)
                if first_seg["distance"] <= self._sensor_range:
                    sensor_readings[ray_idx][idx] = \
                        (self._sensor_range - first_seg[
                            "distance"]) / self._sensor_range
        return sensor_readings

    def _get_obs(self):
        wrapped_obs = self.wrapped_env._get_obs()
        if self._top_down_view:
            self.get_top_down_view()

        if self._observe_blocks:
            additional_obs = []
            for block_name, block_type in self.movable_blocks:
                additional_obs.append(
                    self.wrapped_env.get_body_com(block_name))
            wrapped_obs = np.concatenate([wrapped_obs[:3]] + additional_obs +
                                         [wrapped_obs[3:]])

        self.get_range_sensor_obs()
        return wrapped_obs

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, goal=None):
        self.goal = goal

        if self.visualize_goal:  # remove the prev goal and add a new goal
            goal_x, goal_y = goal[0], goal[1]
            size_scaling = self.MAZE_SIZE_SCALING
            # remove the original goal
            try:
                self.worldbody.remove(self.goal_element)
            except AttributeError:
                pass
            # offset all coordinates so that robot starts at the origin
            self.goal_element = \
                ET.SubElement(
                    self.worldbody, "geom",
                    name="goal_%d_%d" % (goal_x, goal_y),
                    pos="%f %f %f" % (goal_x,
                                      goal_y,
                                      self.MAZE_HEIGHT / 2 * size_scaling),
                    # smaller than the block to prevent collision
                    size="%f %f %f" % (0.1 * size_scaling,
                                       0.1 * size_scaling,
                                       self.MAZE_HEIGHT / 2 * size_scaling),
                    type="box",
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba="1.0 0.0 0.0 0.5"
                )
            # Note: running the lines below will make the robot position wrong!
            # (because the graph is rebuilt)
            torso = self.tree.find(".//body[@name='torso']")
            geoms = torso.findall(".//geom")
            for geom in geoms:
                if 'name' not in geom.attrib:
                    raise Exception("Every geom of the torso must have a name "
                                    "defined")
            _, file_path = tempfile.mkstemp(text=True, suffix='.xml')
            self.tree.write(file_path)
            # here we write a temporal file with the robot specifications.
            # Why not the original one??

            model_cls = self.__class__.MODEL_CLASS
            # file to the robot specifications; model_cls is AntEnv
            self.wrapped_env = model_cls(
                *self.args, file_path=file_path, **self.kwargs)

        self.t = 0
        self.trajectory = []
        self.wrapped_env.reset()
        if len(self._init_positions) > 1:
            xy = self._init_positions[self.np_random.randint(
                len(self._init_positions))]
            self.wrapped_env.set_xy(xy)
        return self._get_obs()

    @property
    def viewer(self):
        return self.wrapped_env.viewer

    def render(self, *args, **kwargs):
        return self.wrapped_env.render(*args, **kwargs)

    @property
    def observation_space(self):
        shape = self._get_obs().shape
        high = np.inf * np.ones(shape)
        low = -high
        return gym.spaces.Box(low, high)

    @property
    def action_space(self):
        return self.wrapped_env.action_space

    def _find_robot(self):
        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 'r':
                    return j * size_scaling, i * size_scaling
        assert False, 'No robot in maze specification.'

    def _find_all_robots(self):
        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING
        coords = []
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 'r':
                    coords.append((j * size_scaling, i * size_scaling))
        return coords

    def _is_in_collision(self, pos):
        x, y = pos
        structure = self.MAZE_STRUCTURE
        scale = self.MAZE_SIZE_SCALING
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 1:
                    minx = j * scale - scale * 0.5 - self._init_torso_x
                    maxx = j * scale + scale * 0.5 - self._init_torso_x
                    miny = i * scale - scale * 0.5 - self._init_torso_y
                    maxy = i * scale + scale * 0.5 - self._init_torso_y
                    if minx <= x <= maxx and miny <= y <= maxy:
                        return True
        return False

    def _rowcol_to_xy(self, j, i):
        scale = self.MAZE_SIZE_SCALING
        minx = j * scale - scale * 0.5 - self._init_torso_x
        maxx = j * scale + scale * 0.5 - self._init_torso_x
        miny = i * scale - scale * 0.5 - self._init_torso_y
        maxy = i * scale + scale * 0.5 - self._init_torso_y
        return (minx + maxx) / 2, (miny + maxy) / 2

    def step(self, action):
        self.t += 1
        if self._manual_collision:
            old_pos = self.wrapped_env.get_xy()
            inner_next_obs, inner_reward, inner_done, info = \
                self.wrapped_env.step(action)
            new_pos = self.wrapped_env.get_xy()
            if self._is_in_collision(new_pos):
                self.wrapped_env.set_xy(old_pos)
        else:
            inner_next_obs, inner_reward, inner_done, info = \
                self.wrapped_env.step(action)
        next_obs = self._get_obs()
        done = False
        if self.goal is not None:
            done = bool(((next_obs[:2] - self.goal[:2]) ** 2).sum() < self.EPS)

        new_pos = self.wrapped_env.get_xy()
        if self._is_in_collision(new_pos) or inner_done:
            done = True
        if self.t >= self.max_episode_steps:
            done = True
        return next_obs, inner_reward, done, info
