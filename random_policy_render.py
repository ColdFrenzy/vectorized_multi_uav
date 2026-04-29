import time
import torch
from vmas import make_env
from vmas.simulator.core import Agent
from vmas.simulator.scenario import BaseScenario
from typing import Union
from custom_scenario_utils import get_obstacles_position
# to run the render use this: 
# conda activate vmas 
# LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -s "-screen 0 1400x900x24" python random_policy_render.py


import pyvirtualdisplay
display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
display.start()

def use_vmas_env(
    render: bool,
    num_envs: int,
    n_steps: int,
    device: str,
    scenario: Union[str, BaseScenario],
    continuous_actions: bool,
    **kwargs
):
    """Example function to use a vmas environment.

    This is a simplification of the function in `vmas.examples.use_vmas_env.py`.

    Args:
        continuous_actions (bool): Whether the agents have continuous or discrete actions
        scenario (str, BaseScenario): Name of scenario or scenario class
        device (str): Torch device to use
        render (bool): Whether to render the scenario
        num_envs (int): Number of vectorized environments
        n_steps (int): Number of steps before returning done

    """

    scenario_name = scenario if isinstance(scenario,str) else scenario.__class__.__name__

    env = make_env(
        scenario=scenario,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        seed=0,
        # Environment specific variables
        **kwargs
    )

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0

    for s in range(n_steps):
        step += 1
        print(f"Step {step}")

        actions = []
        for i, agent in enumerate(env.agents):
            action = env.get_random_action(agent)

            actions.append(action)

        obs, rews, dones, info = env.step(actions)

        if render:
            frame = env.render(mode="rgb_array")
            frame_list.append(frame)

    total_time = time.time() - init_time
    print(
        f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device} "
        f"for {scenario_name} scenario."
    )

    if render:
        from moviepy import ImageSequenceClip
        fps=30
        clip = ImageSequenceClip(frame_list, fps=fps)
        clip.write_gif(f'{scenario_name}.gif', fps=fps)


if __name__ == "__main__":
    from vmas_navigation import MultiUAVNavigation
    vmas_device = "cuda" if torch.cuda.is_available() else "cpu"
    n_agents_holonomic = 2
    n_agents_diff_drive = 0
    n_agents_car = 0
    num_envs = 8

    map_string = """
🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
🟩 ⛔  1 🟩 🟩 🟩 🟩 🟩 ⛔ 🟩
🟩 🟩 🟩 🟩 🟩 🟩 🟩 ⛔ 🟩 🟩
🟩 🟩 ⛔ ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩
🟩 🟩 ⛔ ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩
🟩 🟩 ⛔ 🟩 🟩 🟩 🟩 ⛔ 🟩 🟩
🟩 ⛔ ⛔ 🟩 🟩 🟩 🟩 ⛔ 🟩 🟩
🟩 🟩 ⛔ 🟩 ⛔ 🟩 ⛔ ⛔ 🟩 🟩
🟩 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🟩 🟩
🟩 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🟩 🟩
🟩 🟩 a2 🟩 ⛔ 🟩 🟩 a1 🟩 🟩
"""
    world_spawning_x =  map_string.count("\n") - 1 
    world_spawning_y = len(map_string.split("\n")[1].split(" "))
    obstacles_initial_positions = list(get_obstacles_position(map_string, world_spawning_x, world_spawning_y, offset=0.5))

    use_vmas_env(
        render=True,
        num_envs=num_envs,
        n_steps=20,
        device=vmas_device,
        scenario=MultiUAVNavigation(),
        continuous_actions=True,
        # Scenario kwargs
        n_agents_holonomic=n_agents_holonomic,
        n_agents_diff_drive=n_agents_diff_drive,
        n_agents_car=n_agents_car,
        n_obstacles=len(obstacles_initial_positions),
        obstacles_initial_positions=obstacles_initial_positions,
        world_spawning_x = world_spawning_x,
        world_spawning_y = world_spawning_y,
        viewer_size = (world_spawning_x * 100, world_spawning_y * 100), # Size of the viewer window in pixels
        grid_spacing =  1,  # Spacing between grid lines in the rendering
        viewer_zoom = 3, # Zoom level for the viewer
        agent_max_speed = 13,
        lidar_range=2,
        agent_radius=0.4,
    )
    display.stop()