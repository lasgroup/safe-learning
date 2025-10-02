from typing import Callable, Generic, Sequence, Tuple, TypeVar, Union

import flax
import jax
import jax.numpy as jnp
import optax
from absl import logging
from brax.training.agents.sac import checkpoint
from brax.training.replay_buffers import ReplayBuffer
from brax.training.types import Transition

from ss2r.algorithms.sac import pytree_uniform_sampling_queue as pusq
from ss2r.common.wandb import get_wandb_checkpoint

Sample = TypeVar("Sample")
MixType = Union[float, Callable[[int], float], Tuple[float, float, int]]


@flax.struct.dataclass
class RAEReplayBufferState:
    offline_state: pusq.PytreeReplayBufferState
    online_state: pusq.PytreeReplayBufferState
    key: jax.Array
    step: int


class RAEReplayBuffer(ReplayBuffer[RAEReplayBufferState, Sample], Generic[Sample]):
    def __init__(
        self,
        max_replay_size: int,
        dummy_data_sample: Sample,
        sample_batch_size: int,
        wandb_ids: Sequence[str],
        wandb_entity: str | None,
        mix: MixType = 0.5,
    ):
        self.sample_batch_size = sample_batch_size
        self.mix = self._init_mix(mix)
        self.online_buffer = pusq.PytreeUniformSamplingQueue(
            max_replay_size, dummy_data_sample, self.sample_batch_size
        )
        self.offline_buffer: pusq.PytreeUniformSamplingQueue | None = None
        self.wandb_ids = wandb_ids
        self.wandb_entity = wandb_entity
        self._dummy_data_sample = dummy_data_sample

    def init(self, key: jax.Array) -> RAEReplayBufferState:
        """Initialize both buffers and load offline data from disk."""
        key_online, key_next = jax.random.split(key, 2)
        online_state = self.online_buffer.init(key_online)
        offline_state = prepare_offline_data(self.wandb_ids, self.wandb_entity)
        max_size = offline_state.data.reward.shape[0]
        self.offline_buffer = pusq.PytreeUniformSamplingQueue(
            max_size, self._dummy_data_sample, self.sample_batch_size
        )
        logging.info("Restoring replay buffer state with %d samples", max_size)
        return RAEReplayBufferState(
            online_state=online_state,  # type: ignore
            offline_state=offline_state,  # type: ignore
            key=key_next,
            step=0,
        )

    def _init_mix(self, mix: MixType) -> Callable[[int], float]:
        if isinstance(mix, float):
            return lambda step: mix  # type: ignore
        elif isinstance(mix, Sequence) and len(mix) == 3:
            init_val, end_val, steps = tuple(mix)  # type: ignore
            scheduler = optax.linear_schedule(
                init_value=init_val, end_value=end_val, transition_steps=steps
            )
            return lambda step: scheduler(step)  # type: ignore
        elif callable(mix):
            return lambda step: mix(step)  # type: ignore
        else:
            raise ValueError(
                "mix must be a float, a scheduler function, or a (init, end, steps) tuple."
            )

    def insert_internal(
        self, state: RAEReplayBufferState, samples: Sample
    ) -> RAEReplayBufferState:
        """Insert new online samples."""
        new_online_state = self.online_buffer.insert(state.online_state, samples)
        new_offline_state = state.offline_state
        return state.replace(  # type: ignore
            online_state=new_online_state,
            offline_state=new_offline_state,
        )

    def sample_internal(
        self, state: RAEReplayBufferState
    ) -> Tuple[RAEReplayBufferState, Sample]:
        """Sample from both buffers and return merged batch."""
        assert self.offline_buffer is not None
        mix_value = self.mix(state.step)
        key_online, key_offline, new_key = jax.random.split(state.key, 3)
        online_state, online_samples = self.online_buffer.sample(
            state.online_state.replace(key=key_online)  # type: ignore
        )
        offline_state, offline_samples = self.offline_buffer.sample(
            state.offline_state.replace(key=key_offline)  # type: ignore
        )
        combined_samples = jax.tree_util.tree_map(
            lambda o, f: jnp.concatenate([o, f]), online_samples, offline_samples
        )
        online_weight = jnp.full((self.sample_batch_size,), mix_value)
        offline_weight = jnp.full((self.sample_batch_size,), 1.0 - mix_value)
        probs = jnp.concatenate([online_weight, offline_weight])
        new_key, subkey = jax.random.split(new_key)
        indices = jax.random.choice(
            subkey,
            2 * self.sample_batch_size,
            shape=(self.sample_batch_size,),
            p=probs,
            replace=True,
        )
        combined_samples = jax.tree_util.tree_map(
            lambda s: jnp.take(s, indices, axis=0), combined_samples
        )
        new_state = state.replace(  # type: ignore
            online_state=online_state,
            offline_state=offline_state,
            key=new_key,
            step=state.step + 1,
        )
        return new_state, combined_samples

    def size(self, state: RAEReplayBufferState) -> int:
        assert self.offline_buffer is not None
        return self.online_buffer.size(state.online_state) + self.offline_buffer.size(
            state.offline_state
        )


def prepare_offline_data(wandb_ids, wandb_entity):
    data = []
    last_position = 0
    for wandb_id in wandb_ids:
        checkpoint_path = get_wandb_checkpoint(wandb_id, wandb_entity)
        params = checkpoint.load(checkpoint_path)
        replay_buffer_state = params[-1]
        size = (
            replay_buffer_state["insert_position"]
            - replay_buffer_state["sample_position"]
        )
        valid_data = jax.tree.map(
            lambda x: x[
                replay_buffer_state["sample_position"] : replay_buffer_state[
                    "insert_position"
                ]
            ],
            replay_buffer_state["data"],
        )
        data.append(valid_data)
        last_position += size
    concatenated_data = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *data)
    concatenated_data = Transition(**concatenated_data)
    key = replay_buffer_state["key"]
    assert last_position == (concatenated_data.shape[0] - 1)
    return pusq.PytreeReplayBufferState(
        data=concatenated_data,
        insert_position=jnp.array(last_position, dtype=jnp.int32),
        key=key,
        sample_position=jnp.array(0, dtype=jnp.int32),
    )
