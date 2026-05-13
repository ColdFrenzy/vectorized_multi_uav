from vmas.simulator.utils import ScenarioUtils
from typing import Dict, List, Sequence, Tuple, Union
from torch import Tensor, device
import torch
import warnings



class CustomScenarioUtils(ScenarioUtils):

    @staticmethod
    def spawn_entities_fixed_obstacles(
        entities,
        obstacles,
        obstacles_positions: List[Tensor],
        world,
        env_index: int,
        min_dist_between_entities: float,
        x_bounds: Tuple[int, int],
        y_bounds: Tuple[int, int],
        walls: List[Tensor],
        wall_configs: List[Dict],
        occupied_positions: Tensor = None,
        disable_warn: bool = False,
    ):
        batch_size = world.batch_dim if env_index is None else 1

        if occupied_positions is None:
            occupied_positions = torch.zeros(
                (batch_size, 0, world.dim_p), device=world.device
            )
        for i, obstacle in enumerate(obstacles):
            pos = torch.tensor(list(obstacles_positions[i]), device=world.device, dtype=torch.float32).view(1, 1, -1).repeat(batch_size, 1, 1) # pos shape: (batch_size, 1, dim_p)
            occupied_positions = torch.cat([occupied_positions, pos], dim=1) # occupied_positions shape: (batch_size, num_entities, dim_p)
            obstacle.set_pos(pos.squeeze(1), batch_index=env_index)
        
        for wall, config in zip(walls, wall_configs):
            wall_pos = torch.tensor(config["pos"], device=world.device, dtype=torch.float32).view(1, 1, -1).repeat(batch_size, 1, 1) # wall_pos shape: (batch_size, 1, dim_p)
            wall_rot = torch.tensor([config["rot"]], device=world.device, dtype=torch.float32).view(1, 1, -1).repeat(batch_size, 1, 1) # wall_rot shape: (batch_size, 1, 1)
            occupied_positions = torch.cat([occupied_positions, wall_pos], dim=1) # occupied_positions shape: (batch_size, num_entities, dim_p)
            wall.set_pos(wall_pos.squeeze(1), batch_index=env_index)
            wall.set_rot(wall_rot.squeeze(1), batch_index=env_index)

        for entity in entities:
            if entity.name.startswith("goal"):
                pos = CustomScenarioUtils.find_random_int_pos_for_entity(
                    occupied_positions,
                    env_index,
                    world,
                    min_dist_between_entities,
                    x_bounds,
                    y_bounds,
                    disable_warn,
                )
            else:
                pos = ScenarioUtils.find_random_pos_for_entity(
                occupied_positions,
                env_index,
                world,
                min_dist_between_entities,
                x_bounds,
                y_bounds,
                disable_warn,
            ) # pos shape: (num_env, 1, dim_p)
            occupied_positions = torch.cat([occupied_positions, pos], dim=1)
            entity.set_pos(pos.squeeze(1), batch_index=env_index)

    @staticmethod
    def find_random_int_pos_for_entity(                
        occupied_positions: torch.Tensor,
        env_index: int,
        world,
        min_dist_between_entities: float,
        x_bounds: Tuple[int, int],
        y_bounds: Tuple[int, int],
        disable_warn: bool = False,
    ):
        batch_size = world.batch_dim if env_index is None else 1

        pos = None
        tries = 0
        while True:
            # Shift integer grid picks to centered cell coordinates inside bounds.
            proposed_pos = torch.cat(
                [
                    torch.empty(
                        (batch_size, 1, 1),
                        device=world.device,
                        dtype=torch.float32,
                    ).random_(*x_bounds) + 0.5,
                    torch.empty(
                        (batch_size, 1, 1),
                        device=world.device,
                        dtype=torch.float32,
                    ).random_(*y_bounds) + 0.5,
                ],
                dim=2,
            )
            if pos is None:
                pos = proposed_pos
            if occupied_positions.shape[1] == 0:
                break

            dist = torch.cdist(occupied_positions, pos)
            overlaps = torch.any((dist < min_dist_between_entities).squeeze(2), dim=1)
            if torch.any(overlaps, dim=0):
                pos[overlaps] = proposed_pos[overlaps]
            else:
                break
            tries += 1
            if tries > 50_000 and not disable_warn:
                warnings.warn(
                    "It is taking many iterations to spawn the entity, make sure the bounds or "
                    "the min_dist_between_entities are not too tight to fit all entities."
                    "You can disable this warning by setting disable_warn=True"
                )
        return pos

def get_obstacles_position(map, rows, cols, offset):
    obstacles_positions = []

    for r, line in enumerate(map.strip().split("\n")):
        # goal state have double space before or after '1', so replace double spaces with single space
        for c, cell in enumerate(line.split()):
            if cell == "⛔":
                obstacles_positions.append(((c+offset) - int(cols/2), int(rows/2)-(r + offset)))

    return obstacles_positions