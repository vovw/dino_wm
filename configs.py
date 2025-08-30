"""
Simple configuration classes to replace Hydra configs.
Only includes configurations for PushT and granular (deformable) environments.
"""
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class EncoderConfig:
    _target_: str = "models.dino.DinoEncoder"
    model_name: str = "dinov2_vitb14"
    freeze: bool = True
    norm_layer: bool = True


@dataclass
class ActionEncoderConfig:
    _target_: str = "models.proprio.ProprioEncoder"


@dataclass
class ProprioEncoderConfig:
    _target_: str = "models.proprio.ProprioEncoder"


@dataclass
class DecoderConfig:
    _target_: str = "models.vqvae.VQVAEDecoder"
    n_embed: int = 1024
    embed_dim: int = 256
    n_down: int = 2


@dataclass
class PredictorConfig:
    _target_: str = "models.vit.ViT"
    patch_size: int = 1
    depth: int = 4
    heads: int = 8
    mlp_dim: int = 512
    dropout: float = 0.1
    emb_dropout: float = 0.1


@dataclass
class TransformConfig:
    _target_: str = "datasets.img_transforms.default_transform"
    img_size: int = 224


@dataclass
class PushTDatasetConfig:
    _target_: str = "datasets.pusht_dset.load_pusht_slice_train_val"
    with_velocity: bool = True
    n_rollout: Optional[int] = None
    normalize_action: bool = True
    data_path: str = ""  # Will be set from environment variable
    split_ratio: float = 0.9
    transform: TransformConfig = None

    def __post_init__(self):
        if self.transform is None:
            self.transform = TransformConfig()
        if not self.data_path:
            self.data_path = os.environ.get("DATASET_DIR", "./data") + "/pusht_noise"


@dataclass
class GranularDatasetConfig:
    _target_: str = "datasets.deformable_env_dset.load_deformable_dset_slice_train_val"
    n_rollout: Optional[int] = None
    normalize_action: bool = True
    data_path: str = ""  # Will be set from environment variable
    object_name: str = "granular"
    split_ratio: float = 0.9
    transform: TransformConfig = None

    def __post_init__(self):
        if self.transform is None:
            self.transform = TransformConfig()
        if not self.data_path:
            self.data_path = os.environ.get("DATASET_DIR", "./data") + "/deformable"


@dataclass
class PushTEnvConfig:
    name: str = "pusht"
    args: List = None
    kwargs: Dict[str, Any] = None
    dataset: PushTDatasetConfig = None
    decoder_path: Optional[str] = None
    num_workers: int = 16

    def __post_init__(self):
        if self.args is None:
            self.args = []
        if self.kwargs is None:
            self.kwargs = {"with_velocity": True, "with_target": True}
        if self.dataset is None:
            self.dataset = PushTDatasetConfig()


@dataclass
class GranularEnvConfig:
    name: str = "deformable_env"
    args: List = None
    kwargs: Dict[str, Any] = None
    dataset: GranularDatasetConfig = None
    decoder_path: Optional[str] = None
    num_workers: int = 16

    def __post_init__(self):
        if self.args is None:
            self.args = []
        if self.kwargs is None:
            self.kwargs = {"object_name": "granular"}
        if self.dataset is None:
            self.dataset = GranularDatasetConfig()


@dataclass
class TrainingConfig:
    seed: int = 0
    epochs: int = 100
    batch_size: int = 32
    save_every_x_epoch: int = 1
    reconstruct_every_x_batch: int = 500
    num_reconstruct_samples: int = 6
    encoder_lr: float = 1e-6
    decoder_lr: float = 3e-4
    predictor_lr: float = 5e-4
    action_encoder_lr: float = 5e-4


@dataclass
class ModelConfig:
    _target_: str = "models.visual_world_model.VWorldModel"
    train_encoder: bool = False
    train_predictor: bool = True
    train_decoder: bool = True


@dataclass
class TrainConfig:
    # Environment configuration
    env: Any = None
    
    # Model components
    encoder: EncoderConfig = None
    action_encoder: ActionEncoderConfig = None
    proprio_encoder: ProprioEncoderConfig = None
    decoder: DecoderConfig = None
    predictor: PredictorConfig = None
    model: ModelConfig = None
    
    # Training settings
    training: TrainingConfig = None
    
    # Base configuration
    ckpt_base_path: str = "./"
    img_size: int = 224
    frameskip: int = 5
    concat_dim: int = 1
    normalize_action: bool = True
    
    # Action encoder settings
    action_emb_dim: int = 10
    num_action_repeat: int = 1
    
    # Proprio encoder settings
    proprio_emb_dim: int = 10
    num_proprio_repeat: int = 1
    
    # Sequence settings
    num_hist: int = 3
    num_pred: int = 1
    has_predictor: bool = True
    has_decoder: bool = True
    
    # Debug
    debug: bool = False
    
    def __post_init__(self):
        if self.encoder is None:
            self.encoder = EncoderConfig()
        if self.action_encoder is None:
            self.action_encoder = ActionEncoderConfig()
        if self.proprio_encoder is None:
            self.proprio_encoder = ProprioEncoderConfig()
        if self.decoder is None:
            self.decoder = DecoderConfig()
        if self.predictor is None:
            self.predictor = PredictorConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()


@dataclass
class ObjectiveConfig:
    _target_: str = "planning.objectives.create_objective_fn"
    alpha: float = 1.0
    base: float = 2.0
    mode: str = "last"


@dataclass
class CEMPlannerConfig:
    _target_: str = "planning.cem.CEMPlanner"
    horizon: int = 5
    topk: int = 30
    num_samples: int = 300
    var_scale: float = 1.0
    opt_steps: int = 30
    eval_every: int = 1


@dataclass
class MPCPlannerConfig:
    _target_: str = "planning.mpc.MPCPlanner"
    max_iter: Optional[int] = None
    n_taken_actions: int = 5
    sub_planner: CEMPlannerConfig = None
    name: str = "mpc_cem"

    def __post_init__(self):
        if self.sub_planner is None:
            self.sub_planner = CEMPlannerConfig()


@dataclass
class PlanConfig:
    # Model to load for planning
    ckpt_base_path: str = "./"
    model_name: Optional[str] = None
    model_epoch: str = "latest"
    
    # Planning settings
    seed: int = 99
    n_evals: int = 50
    goal_source: str = "dset"  # 'random_state' or 'dset' or 'random_action'
    goal_H: int = 5
    n_plot_samples: int = 10
    
    # Debug
    debug_dset_init: bool = False
    wandb_logging: bool = True
    
    # Planning components
    objective: ObjectiveConfig = None
    planner: MPCPlannerConfig = None
    
    def __post_init__(self):
        if self.objective is None:
            self.objective = ObjectiveConfig()
        if self.planner is None:
            self.planner = MPCPlannerConfig()


def get_pusht_train_config() -> TrainConfig:
    """Get training configuration for PushT environment."""
    config = TrainConfig()
    config.env = PushTEnvConfig()
    return config


def get_granular_train_config() -> TrainConfig:
    """Get training configuration for granular environment."""
    config = TrainConfig()
    config.env = GranularEnvConfig()
    return config


def get_pusht_plan_config() -> PlanConfig:
    """Get planning configuration for PushT environment."""
    config = PlanConfig()
    config.n_evals = 50
    config.goal_source = "dset"
    config.planner.sub_planner.opt_steps = 30
    return config


def get_granular_plan_config() -> PlanConfig:
    """Get planning configuration for granular environment."""
    config = PlanConfig()
    config.n_evals = 50
    config.goal_source = "dset"  # or "random_state" depending on preference
    config.planner.sub_planner.opt_steps = 30
    return config
