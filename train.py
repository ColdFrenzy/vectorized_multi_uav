import copy
import torch
from typing import Callable, Optional
from benchmarl.environments import VmasTask
from benchmarl.utils import DEVICE_TYPING
from torchrl.envs import EnvBase, VmasEnv
from vmas_navigation import MultiUAVNavigation
from benchmarl.experiment import ExperimentConfig
from benchmarl.algorithms import MappoConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.experiment import Experiment
from custom_scenario_utils import get_obstacles_position

def get_env_fun(
    self,
    num_envs: int,
    continuous_actions: bool,
    seed: Optional[int],
    device: DEVICE_TYPING,
) -> Callable[[], EnvBase]:
    config = copy.deepcopy(self.config)
    if (hasattr(self, "name") and self.name is "NAVIGATION") or (
        self is VmasTask.NAVIGATION
    ):  # We substitute the original vmas scenario with our custom one
        scenario = MultiUAVNavigation()  # .... ends here
    else:
        scenario = self.name.lower()
    return lambda: VmasEnv(
        scenario=scenario,
        num_envs=num_envs,
        continuous_actions=continuous_actions,
        seed=seed,
        device=device,
        categorical_actions=True,
        clamp_actions=True,
        **config,
    )

try:
    from benchmarl.environments import VmasClass
    VmasClass.get_env_fun = get_env_fun
except ImportError:
    VmasTask.get_env_fun = get_env_fun

#####################
# EXPERIMENT CONFIG #
#####################
train_device = "cuda" if torch.cuda.is_available() else "cpu"
vmas_device = "cuda" if torch.cuda.is_available() else "cpu"
# Loads from "benchmarl/conf/experiment/base_experiment.yaml"
experiment_config = ExperimentConfig.get_from_yaml() # We start by loading the defaults
# Override devices
experiment_config.sampling_device = vmas_device
experiment_config.train_device = train_device
experiment_config.max_n_frames = 10_000 # 10_000_000 # Number of frames before training ends
experiment_config.gamma = 0.99
experiment_config.on_policy_collected_frames_per_batch = 2000 # 60_000 # Number of frames collected each iteration
experiment_config.on_policy_n_envs_per_worker = 5 # 600 # Number of vmas vectorized enviornemnts (each will collect 100 steps, see max_steps in task_config -> 600 * 100 = 60_000 the number above)
experiment_config.on_policy_n_minibatch_iters = 10 # 45
experiment_config.on_policy_minibatch_size = 256 # 4096
experiment_config.evaluation = True
experiment_config.render = True
experiment_config.share_policy_params = True # Policy parameter sharing on
experiment_config.evaluation_interval = 4000 # 120_000 # Interval in terms of frames, will evaluate every 120_000 / 60_000 = 2 iterations
experiment_config.evaluation_episodes = 20 # 200 # Number of vmas vectorized enviornemnts used in evaluation
experiment_config.loggers = ["csv"] #wandb

###############
# TASK CONFIG #
###############
task = VmasTask.NAVIGATION.get_from_yaml()
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

task.config = {
        "max_steps": 100,
        "n_agents_holonomic": 2,
        "n_agents_diff_drive": 0,
        "n_agents_car": 0,
        "lidar_range": 2,
        "comms_rendering_range": 0,
        "shared_rew": False,
        "n_obstacles": len(obstacles_initial_positions),
        "obstacles_initial_positions": obstacles_initial_positions,
        "world_spawning_x": world_spawning_x,
        "world_spawning_y": world_spawning_y,
        "viewer_size": (world_spawning_x * 100, world_spawning_y * 100), # Size of the viewer window in pixels
        "grid_spacing":  1,  # Spacing between grid lines in the rendering
        "viewer_zoom": 3, # Zoom level for the viewer
        "agent_max_speed": 13,
        "agent_radius": 0.4,
}

####################
# ALGORITHM CONFIG #
####################
# We can load from "benchmarl/conf/algorithm/mappo.yaml"
algorithm_config = MappoConfig.get_from_yaml()
# Or create it from scratch
algorithm_config = MappoConfig(
        share_param_critic=True, # Critic param sharing on
        clip_epsilon=0.2,
        entropy_coef=0.001, # We modify this, default is 0
        critic_coef=1,
        loss_critic_type="l2",
        lmbda=0.9,
        scale_mapping="biased_softplus_1.0", # Mapping for standard deviation
        use_tanh_normal=True,
        minibatch_advantage=False,
    )

################
# MODEL CONFIG #
################
model_config = MlpConfig(
        num_cells=[256, 256], # Two layers with 256 neurons each
        layer_class=torch.nn.Linear,
        activation_class=torch.nn.Tanh,
    )

# Loads from "benchmarl/conf/model/layers/mlp.yaml" (in this case we use the defaults so it is the same)
model_config = MlpConfig.get_from_yaml()
critic_model_config = MlpConfig.get_from_yaml()


##############
# EXPERIMENT #
##############
experiment = Experiment(
    task=task,
    algorithm_config=algorithm_config,
    model_config=model_config,
    critic_model_config=critic_model_config,
    seed=0,
    config=experiment_config,
)
experiment.run()
# conda activate vmas 
# LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -s "-screen 0 1400x900x24" python train.py
