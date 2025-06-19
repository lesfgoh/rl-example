import numpy as np
from enum import Flag, Enum
from environment.types import Direction
from itertools import product


class TileType(Enum):
    NO_VISION = 0
    EMPTY = 1
    RECON = 2
    MISSION = 3


class TileWalls(Flag):
    RIGHT_WALL = 2**4
    BOTTOM_WALL = 2**5
    LEFT_WALL = 2**6
    TOP_WALL = 2**7


class TileAgent(Flag):
    SCOUT = 2**2
    GUARD = 2**3


class WorldMap:
    tile_dtype = np.dtype(TileType, TileWalls, TileAgent)

    # lookup table: row delta, col delta, current tile wall and what
    # the neighbor wall should be changed to if present
    tile_neighbors = np.array(
        [
            [0, -1, TileWalls.LEFT_WALL, TileWalls.RIGHT_WALL],  # left
            [-1, 0, TileWalls.TOP_WALL, TileWalls.BOTTOM_WALL],  # up
            [0, 1, TileWalls.RIGHT_WALL, TileWalls.LEFT_WALL],  # right
            [1, 0, TileWalls.BOTTOM_WALL, TileWalls.TOP_WALL],  # down
        ]
    )
    viewcone_pos = list(product(range(7), range(5)))
    rot_viewcone_pos = list(product(range(5), range(7)))

    def __init__(self):
        self.map = np.zeros((16, 16), dtype=self.tile_dtype)

        for x, y in product(range(16), range(16)):
            self.map[x][y] = [
                TileType.NO_VISION,
                TileWalls(0),
                TileAgent(0),
            ]

        self.decay = np.zeros((16, 16))

    def convert_to_tile(self, packed_tile):
        packed_tile = int(packed_tile)
        return [
            TileType(packed_tile & 0b11),
            TileWalls(packed_tile & 0b11110000),
            TileAgent(packed_tile & 0b1100),
        ]

    def flip_walls_udlr(self, walls):
        # tiles are mirrored horizontally and vertically

        new_walls = TileWalls(0)

        if TileWalls.LEFT_WALL in walls:
            new_walls |= TileWalls.RIGHT_WALL

        if TileWalls.RIGHT_WALL in walls:
            new_walls |= TileWalls.LEFT_WALL

        if TileWalls.BOTTOM_WALL in walls:
            new_walls |= TileWalls.TOP_WALL

        if TileWalls.TOP_WALL in walls:
            new_walls |= TileWalls.BOTTOM_WALL

        return new_walls

    def rotate_walls_cw(self, walls):
        new_walls = TileWalls(0)

        if TileWalls.BOTTOM_WALL in walls:
            new_walls |= TileWalls.LEFT_WALL

        if TileWalls.LEFT_WALL in walls:
            new_walls |= TileWalls.TOP_WALL

        if TileWalls.TOP_WALL in walls:
            new_walls |= TileWalls.RIGHT_WALL

        if TileWalls.RIGHT_WALL in walls:
            new_walls |= TileWalls.BOTTOM_WALL

        return new_walls

    def rotate_walls_ccw(self, walls):
        new_walls = TileWalls(0)

        if TileWalls.BOTTOM_WALL in walls:
            new_walls |= TileWalls.RIGHT_WALL

        if TileWalls.LEFT_WALL in walls:
            new_walls |= TileWalls.BOTTOM_WALL

        if TileWalls.TOP_WALL in walls:
            new_walls |= TileWalls.LEFT_WALL

        if TileWalls.RIGHT_WALL in walls:
            new_walls |= TileWalls.TOP_WALL

        return new_walls

    def unpack_viewcone(self, viewcone):
        viewcone_shape = np.shape(viewcone)

        unpacked = np.zeros(
            (viewcone_shape[0], viewcone_shape[1]), dtype=self.tile_dtype
        )

        for row_idx, row in enumerate(viewcone):
            for tile_idx, tile in enumerate(row):
                unpacked[row_idx][tile_idx] = self.convert_to_tile(tile)

        return unpacked

    def check_bounds(self, row, col):
        return 0 <= row < 16 and 0 <= col < 16

    def compare_against_state(self, state_map):
        for row_idx in range(16):
            for col_idx in range(16):
                tile = self.map[row_idx, col_idx]
                if tile[0] == TileType.NO_VISION:
                    continue
                if tile != state_map.map[row_idx, col_idx]:
                    print(
                        f"Our state diffs at ({row_idx, col_idx}). Ours {self.map[row_idx, col_idx]}, theirs {state_map.map[row_idx, col_idx]}"
                    )

    def state_onto_map(self, state):
        """
        Destructive action, will overwrite current internal map
        """

        for row_idx in range(16):
            for col_idx in range(16):
                self.map[row_idx, col_idx] = self.convert_to_tile(
                    state[row_idx, col_idx]
                )

    def update_from_viewcone(
        self,
        agent_row,
        agent_col,
        unpacked_viewcone,
        tile_rotate_fun,
        viewcone_iter,
        row_shift=-2,
        col_shift=-2,
    ):
        for row, col in viewcone_iter:
            tile = unpacked_viewcone[row, col]

            # if the agent has no vision, it either means
            # something is blocking or out of map
            if tile[0] == TileType.NO_VISION:
                continue

            if tile_rotate_fun:
                tile[1] = tile_rotate_fun(tile[1])

            new_row, new_col = (
                agent_row + row + row_shift,
                agent_col + col + col_shift,
            )
            self.map[new_row, new_col][1] |= tile[1]

            for n_row, n_col, cur_tile_check, n_tile_add in self.tile_neighbors:
                if (
                    not self.check_bounds(n_row, n_col)
                    or cur_tile_check not in self.map[n_row, n_col]
                ):
                    continue

                self.map[n_row, n_col][1] |= n_tile_add

            new_walls = self.map[new_row, new_col][1] | tile[1]
            self.map[new_row, new_col] = [tile[0], new_walls, tile[2]]

    def viewcone_onto_map(self, agent_loc, agent_dir, viewcone):
        unpacked_viewcone = self.unpack_viewcone(viewcone)

        # convert cartesian coordinates to matrix coordinates
        agent_col, agent_row = agent_loc
        agent_dir = Direction(agent_dir)

        explored = set()

        for row, col in self.viewcone_pos:
            if unpacked_viewcone[row, col] == TileType.NO_VISION:
                continue

            explored.add((row, col))

        if agent_dir in set([Direction.DOWN, Direction.LEFT, Direction.RIGHT]):
            unpacked_viewcone = np.fliplr(unpacked_viewcone)

        match agent_dir:
            case Direction.RIGHT:
                # rotate 90 CCW
                unpacked_viewcone = np.rot90(unpacked_viewcone, axes=(0, 1))
                self.update_from_viewcone(
                    agent_row,
                    agent_col,
                    unpacked_viewcone,
                    None,
                    self.rot_viewcone_pos,
                )

            case Direction.DOWN:
                self.update_from_viewcone(
                    agent_row,
                    agent_col,
                    unpacked_viewcone,
                    self.rotate_walls_cw,
                    self.viewcone_pos,
                )

            case Direction.LEFT:
                # rotate 90 CW
                unpacked_viewcone = np.rot90(unpacked_viewcone, axes=(1, 0))
                self.update_from_viewcone(
                    agent_row,
                    agent_col,
                    unpacked_viewcone,
                    self.flip_walls_udlr,
                    self.rot_viewcone_pos,
                    col_shift=-4,
                )

            case Direction.UP:
                unpacked_viewcone = np.flipud(unpacked_viewcone)
                self.update_from_viewcone(
                    agent_row,
                    agent_col,
                    unpacked_viewcone,
                    self.rotate_walls_ccw,
                    self.viewcone_pos,
                    row_shift=-4,
                )

    def print_ascii_map(self):
        ascii_map = "-=" * 20 + "\n"

        for i in self.map:
            row = ""
            for j in i:
                if TileType.NO_VISION == j[0]:
                    row += "X "
                elif TileAgent.SCOUT in j[2]:
                    row += "O "
                else:
                    row += "= "

            ascii_map += row + "\n"

        print(ascii_map + "-=" * 20)
