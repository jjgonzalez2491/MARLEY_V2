import gymnasium as gym
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch, try_import_tf
from ray.rllib.utils.torch_utils import FLOAT_MIN

torch, nn = try_import_torch()
tf1, tf, tfv = try_import_tf()


class ActionMaskRLMBase(RLModule):
    """Base class for action-masking RLModules, compatible with RLlib 2.35+.

    In the new API stack, RLModule.__init__ receives observation_space,
    action_space, model_config, etc. directly as keyword arguments instead
    of a single RLModuleConfig object. This class intercepts the
    observation_space kwarg to strip the action_mask key before passing it
    to the parent PPO module, which only expects the 'observations' subspace.
    """

    def __init__(self, observation_space=None, **kwargs):
        if (
            isinstance(observation_space, gym.spaces.Dict)
            and "observations" in observation_space.spaces
        ):
            inner_obs_space = observation_space["observations"]
        else:
            inner_obs_space = observation_space

        self._full_obs_space = observation_space
        super().__init__(observation_space=inner_obs_space, **kwargs)


class TorchActionMaskRLM(ActionMaskRLMBase, PPOTorchRLModule):
    def _forward_inference(self, batch, **kwargs):
        return mask_forward_fn_torch(super()._forward_inference, batch, **kwargs)

    def _forward_train(self, batch, *args, **kwargs):
        return mask_forward_fn_torch(super()._forward_train, batch, **kwargs)

    def _forward_exploration(self, batch, *args, **kwargs):
        return mask_forward_fn_torch(super()._forward_exploration, batch, **kwargs)

    def compute_values(self, batch, embeddings=None):
        # compute_values is called directly by the learner connector during GAE
        # computation and bypasses _forward_*.  At this point batch["obs"] still
        # holds the full {"action_mask": ..., "observations": ...} dict, so we
        # must strip it here before the parent's critic encoder runs.
        batch = _strip_action_mask_from_batch(batch)
        return super().compute_values(batch, embeddings=embeddings)


# TF support: only define if the TF PPO module is available in this version.
try:
    from ray.rllib.algorithms.ppo.tf.ppo_tf_rl_module import PPOTfRLModule

    class TFActionMaskRLM(ActionMaskRLMBase, PPOTfRLModule):
        def _forward_inference(self, batch, **kwargs):
            return mask_forward_fn_tf(super()._forward_inference, batch, **kwargs)

        def _forward_train(self, batch, *args, **kwargs):
            return mask_forward_fn_tf(super()._forward_train, batch, **kwargs)

        def _forward_exploration(self, batch, *args, **kwargs):
            return mask_forward_fn_tf(super()._forward_exploration, batch, **kwargs)

except ImportError:
    pass


def _strip_action_mask_from_batch(batch):
    """Return a shallow-copied batch with batch['obs'] replaced by the inner
    'observations' tensor, stripping the action_mask.  Safe to call even if
    the obs is already a plain tensor (no-op in that case).
    """
    obs = batch.get(SampleBatch.OBS, None)
    if not isinstance(obs, dict) or "observations" not in obs:
        # Already stripped or not a masked obs — nothing to do.
        return batch

    inner_obs = obs["observations"]
    if isinstance(inner_obs, dict):
        # Nested dict obs: concatenate all leaf tensors into one flat tensor.
        inner_obs = torch.cat(
            [
                v.float().reshape(v.shape[0], -1) if isinstance(v, torch.Tensor)
                else torch.tensor(v, dtype=torch.float32)
                for v in inner_obs.values()
            ],
            dim=-1,
        )
    elif isinstance(inner_obs, torch.Tensor):
        inner_obs = inner_obs.float()

    # Shallow-copy the batch so we don't mutate the original in-place.
    new_batch = {**batch}
    new_batch[SampleBatch.OBS] = inner_obs
    return new_batch


def mask_forward_fn_torch(forward_fn, batch, **kwargs):
    _check_batch(batch)

    obs_dict = batch[SampleBatch.OBS]

    # Cast action_mask to float32: torch.log(0.0) = -inf in float16 overflows
    # FLOAT_MIN (a float32 constant), causing a clamp error.
    action_mask = obs_dict["action_mask"]
    if isinstance(action_mask, torch.Tensor):
        action_mask = action_mask.float()

    # Strip the action mask and replace batch["obs"] with a plain float32 tensor
    # so the parent PPO encoder's Linear layers receive the correct input type.
    batch = _strip_action_mask_from_batch(batch)

    outputs = forward_fn(batch, **kwargs)

    # Apply the action mask: invalid actions get logit → -inf (prob → 0).
    logits = outputs[SampleBatch.ACTION_DIST_INPUTS]
    inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
    outputs[SampleBatch.ACTION_DIST_INPUTS] = logits + inf_mask

    return outputs


def mask_forward_fn_tf(forward_fn, batch, **kwargs):
    _check_batch(batch)

    action_mask = batch[SampleBatch.OBS]["action_mask"]
    batch[SampleBatch.OBS] = batch[SampleBatch.OBS]["observations"]

    outputs = forward_fn(batch, **kwargs)

    logits = outputs[SampleBatch.ACTION_DIST_INPUTS]
    inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
    outputs[SampleBatch.ACTION_DIST_INPUTS] = logits + inf_mask

    return outputs


def _check_batch(batch):
    """Validate that the batch contains the required action-masking keys."""
    obs = batch[SampleBatch.OBS]
    if not isinstance(obs, dict) or "action_mask" not in obs:
        raise ValueError(
            "Action mask not found in observation. This RLModule requires "
            "the environment to return a Dict observation space of the form:\n"
            "  {'action_mask': Box(0.0, 1.0, shape=(n_actions,)),\n"
            "   'observations': <your_obs_space>}"
        )
    if "observations" not in obs:
        raise ValueError(
            "'observations' key not found in observation dict. "
            "Expected observation space:\n"
            "  {'action_mask': Box(0.0, 1.0, shape=(n_actions,)),\n"
            "   'observations': <your_obs_space>}"
        )