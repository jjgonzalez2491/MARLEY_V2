"""
action_mask_lstm.py
───────────────────
"""

import gymnasium as gym
import numpy as np
from dataclasses import dataclass, field

from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.core.models.specs.specs_base import TensorSpec
from ray.rllib.core.models.base import Encoder, ENCODER_OUT
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.configs import ModelConfig, MLPHeadConfig
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


# ── Actor hyper-parameter defaults (override via model_config_dict) ───────────
ACTOR_DEFAULTS = {
    "fcnet_hiddens":    [512, 512],   # residual trunk hidden layers (shared with value head)
    "head_hidden":      64,           # hidden size of each per-action-dim head
    "fcnet_activation": "relu",       # activation used throughout
}

_ACT = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}


# ─────────────────────────────────────────────────────────────────────────────
# Kept from original file — used when LSTM tokenizer is needed
# ─────────────────────────────────────────────────────────────────────────────

class CustomTorchTokenizer(TorchModel, Encoder):
    def __init__(self, config) -> None:
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)
        self.net = nn.Sequential(
            nn.Linear(config.input_dims[0], config.output_dims[0]),
        )

    def get_output_specs(self):
        output_dim = self.config.output_dims[0]
        return SpecDict(
            {ENCODER_OUT: TensorSpec("b, d", d=output_dim, framework="torch")}
        )

    def _forward(self, inputs: dict, **kwargs):
        return {ENCODER_OUT: self.net(inputs[SampleBatch.OBS])}


@dataclass
class CustomTokenizerConfig(ModelConfig):
    output_dims: tuple = None

    def build(self, framework):
        if framework == "torch":
            return CustomTorchTokenizer(self)


# ─────────────────────────────────────────────────────────────────────────────
# Residual trunk
# ─────────────────────────────────────────────────────────────────────────────

class ResidualTrunk(nn.Module):
    """
    MLP trunk with a residual skip connection between the two hidden layers.

    For trunk_hiddens = [512, 512]:
        Linear(obs_dim → 512) → Act
            │
            ├─ Linear(512 → 512) → Act  ← inner layer
            │        +                  ← residual add (same dim, no projection needed)
            └─────────────────────────►
            │
        output [512]

    If trunk_hiddens has only one layer, no residual is added (nothing to skip).
    If the two middle layers differ in size, a Linear projection is used for the skip.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], activation: str):
        super().__init__()
        act_cls = _ACT[activation.lower()]

        assert len(hidden_dims) >= 1, "trunk_hiddens must have at least one element"

        # First layer: input → hidden[0]
        self.first = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            act_cls(),
        )

        # Residual block: hidden[0] → hidden[1]  (only if len >= 2)
        if len(hidden_dims) >= 2:
            self.residual_layer = nn.Sequential(
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                act_cls(),
            )
            # Projection for skip connection if dims differ
            if hidden_dims[0] != hidden_dims[1]:
                self.skip_proj = nn.Linear(hidden_dims[0], hidden_dims[1], bias=False)
            else:
                self.skip_proj = nn.Identity()
            self.output_dim = hidden_dims[1]
        else:
            self.residual_layer = None
            self.skip_proj      = None
            self.output_dim     = hidden_dims[0]

        # Any remaining layers after the residual block (plain MLP)
        extra_layers = []
        for in_d, out_d in zip(hidden_dims[1:-1], hidden_dims[2:]):
            extra_layers.append(nn.Linear(in_d, out_d))
            extra_layers.append(act_cls())
        self.extra = nn.Sequential(*extra_layers) if extra_layers else None
        if extra_layers:
            self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.first(x)
        if self.residual_layer is not None:
            h = self.residual_layer(h) + self.skip_proj(h)
        if self.extra is not None:
            h = self.extra(h)
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Per-action-dimension head
# ─────────────────────────────────────────────────────────────────────────────

class ActionHead(nn.Module):
    """Small MLP head for one action dimension: [trunk_out → head_hidden → n_acts]."""

    def __init__(self, trunk_out_dim: int, head_hidden: int, n_actions: int,
                 activation: str):
        super().__init__()
        act_cls = _ACT[activation.lower()]
        self.net = nn.Sequential(
            nn.Linear(trunk_out_dim, head_hidden),
            act_cls(),
            nn.Linear(head_hidden, n_actions),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# ─────────────────────────────────────────────────────────────────────────────
# Combined actor network (trunk + heads)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualTrunkMultiHeadActor(TorchModel, Encoder):
    """
    Full actor network: ResidualTrunk + one ActionHead per action dimension.

    The forward pass returns a flat logits tensor of shape [B, sum(nvec)],
    which is what PPOTorchRLModule expects for MultiDiscrete action spaces.
    Action masking is applied downstream in mask_forward_fn_torch — unchanged.

    Parameters sourced from config:
        input_dims[0]   observation dimension
        output_dims[0]  total logit dimension = sum(nvec)  (set by catalog)
        nvec            list of per-dimension action counts
        trunk_hiddens   list of trunk hidden sizes
        head_hidden     int, hidden size of each head
        activation      str
    """

    def __init__(self, config) -> None:
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        obs_dim       = config.input_dims[0]
        nvec          = config.nvec                                    # list[int]
        trunk_hiddens = getattr(config, "trunk_hiddens", ACTOR_DEFAULTS["fcnet_hiddens"])
        head_hidden   = getattr(config, "head_hidden",   ACTOR_DEFAULTS["head_hidden"])
        activation    = getattr(config, "activation",    ACTOR_DEFAULTS["fcnet_activation"])

        # Shared residual trunk
        self.trunk = ResidualTrunk(obs_dim, trunk_hiddens, activation)
        trunk_out  = self.trunk.output_dim

        # One head per action dimension
        self.heads = nn.ModuleList([
            ActionHead(trunk_out, head_hidden, int(n), activation)
            for n in nvec
        ])

    def get_output_specs(self):
        total_logits = self.config.output_dims[0]
        return SpecDict(
            {ENCODER_OUT: TensorSpec("b, d", d=total_logits, framework="torch")}
        )

    def _forward(self, inputs: dict, **kwargs):
        obs = inputs[SampleBatch.OBS]                    # [B, obs_dim]
        h   = self.trunk(obs)                            # [B, trunk_out]

        # Each head produces [B, nvec[i]]; concatenate along dim=-1
        logits = torch.cat([head(h) for head in self.heads], dim=-1)  # [B, sum(nvec)]

        return {ENCODER_OUT: logits}


# ─────────────────────────────────────────────────────────────────────────────
# Model config for the actor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResidualMultiHeadActorConfig(ModelConfig):
    """
    ModelConfig for ResidualTrunkMultiHeadActor.

    Fields set by CustomPPOCatalog.get_action_dist_config():
        input_dims      (obs_dim,)
        output_dims     (sum(nvec),)
        nvec            list of per-dimension action counts

    Fields configurable from training script via model_config_dict:
        fcnet_hiddens, fcnet_activation, head_hidden
    """
    output_dims:    tuple      = None
    nvec:           list       = field(default_factory=list)
    trunk_hiddens:  list       = field(default_factory=lambda: ACTOR_DEFAULTS["fcnet_hiddens"])
    head_hidden:    int        = ACTOR_DEFAULTS["head_hidden"]
    activation:     str        = ACTOR_DEFAULTS["fcnet_activation"]

    def build(self, framework: str):
        if framework == "torch":
            return ResidualTrunkMultiHeadActor(self)
        raise NotImplementedError(f"Framework '{framework}' not supported.")


# ─────────────────────────────────────────────────────────────────────────────
# Custom PPO Catalog
# ─────────────────────────────────────────────────────────────────────────────

class CustomPPOCatalog(PPOCatalog):
    """
    PPO catalog that injects:
      1. ResidualTrunk + per-action-dim heads as the pi (actor) network.
      2. The original CustomTokenizerConfig tokenizer (for LSTM compatibility).

    The value function head is left as PPO's default.

    Hyper-parameters (all optional, read from model_config_dict):
        fcnet_hiddens     list   trunk hidden layers, shared with value head (default: [512, 512])
        fcnet_activation  str    activation, shared with value head          (default: "relu")
        head_hidden       int    hidden size per action-dim head             (default: 64)
    """

    # ── tokenizer (kept from original, used by LSTM encoder) ─────────────────
    @classmethod
    def get_tokenizer_config(
        cls,
        observation_space,
        model_config_dict,
        view_requirements=None,
    ) -> ModelConfig:
        return CustomTokenizerConfig(
            input_dims=observation_space.shape,
            output_dims=(64,),
        )

    # ── pi head: residual trunk + per-action-dim heads ────────────────────────
    def get_action_dist_config(self):
        """
        Override PPO's default flat pi-head with ResidualTrunkMultiHeadActor.

        Introspects self.action_space (a MultiDiscrete) to extract nvec.
        Falls back to the default PPO pi-head for non-MultiDiscrete spaces.
        """
        action_space = self.action_space

        # Only override for MultiDiscrete — leave other spaces to PPO default
        if not isinstance(action_space, gym.spaces.MultiDiscrete):
            return super().get_action_dist_config()

        nvec = action_space.nvec.tolist()           # e.g. [5, 5, 3, 3, ...]
        total_logits = int(np.sum(nvec))

        # Read hyper-parameters from model_config_dict (set in training script)
        # fcnet_hiddens and fcnet_activation are the standard PPO keys — using
        # them here ensures the actor trunk and value head always match.
        mcd           = self.model_config_dict
        trunk_hiddens = mcd.get("fcnet_hiddens",    ACTOR_DEFAULTS["fcnet_hiddens"])
        head_hidden   = mcd.get("head_hidden",      ACTOR_DEFAULTS["head_hidden"])
        activation    = mcd.get("fcnet_activation", ACTOR_DEFAULTS["fcnet_activation"])

        return ResidualMultiHeadActorConfig(
            input_dims    = (self.observation_space.shape[0],),
            output_dims   = (total_logits,),
            nvec          = nvec,
            trunk_hiddens = trunk_hiddens,
            head_hidden   = head_hidden,
            activation    = activation,
        )
