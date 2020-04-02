from gym.envs.registration import register


def reg():
    register(
        id='PointMaze-v0',
        entry_point='mujoco.point_maze_env:PointMazeEnv',
        kwargs={
            "maze_size_scaling": 4,
            "maze_id": "Maze2",
            "maze_height": 0.5,
            "manual_collision": True,
            "goal": (1, 3),
        }
    )

    register(
        id='PointMaze-v1',
        entry_point='mujoco.point_maze_env:PointMazeEnv',
        kwargs={
            "maze_size_scaling": 2,
            "maze_id": "Maze2",
            "maze_height": 0.5,
            "manual_collision": True,
            "goal": (1, 3),
        }
    )
