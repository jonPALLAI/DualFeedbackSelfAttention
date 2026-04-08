"""
Stores completed episodes as variable-length trajectories for padded batch training.
"""

from collections import deque
import numpy as np


class EpisodeReplayBuffer:
    """
    Each episode is a full trajectory: numpy arrays of length L (one row per env step / transition).

    Training samples **entire episodes** (not shuffled single transitions). ``sample_padded_batch``
    builds a batch of shape ``[batch_size, T, ...]`` where each row is one episode from start to
    end, padded after the true length. Use ``max_len=None`` to set ``T`` to the longest episode in
    the batch (no truncation).
    """

    def __init__(self, max_episodes: int = 10000):
        self.episodes = deque(maxlen=max_episodes)

    def __len__(self):
        return len(self.episodes)

    def append_episode(
        self,
        states,
        actions,
        rewards,
        next_states,
        dones,
        state_frames=None,
        next_state_frames=None,
        state_times=None,
        next_state_times=None,
    ):
        ep = {
            'states': np.asarray(states, dtype=np.float32),
            'actions': np.asarray(actions, dtype=np.float32),
            'rewards': np.asarray(rewards, dtype=np.float32),
            'next_states': np.asarray(next_states, dtype=np.float32),
            'dones': np.asarray(dones, dtype=np.bool_),
            'length': len(states),
        }
        if state_frames is not None:
            ep['state_frames'] = np.asarray(state_frames, dtype=np.float16)
            ep['next_state_frames'] = np.asarray(next_state_frames, dtype=np.float16)
        if state_times is not None:
            ep['state_times'] = np.asarray(state_times, dtype=np.int32)
            ep['next_state_times'] = np.asarray(next_state_times, dtype=np.int32)
        self.episodes.append(ep)

    def sample_padded_batch(self, batch_size: int, max_len: int = None, rng: np.random.Generator = None):
        """
        Sample batch_size episodes uniformly, pad to max(batch lengths, capped by max_len).

        Returns:
          states: [B, T_max, D]
          next_states: [B, T_max, D]
          actions, rewards, dones: [B, T_max, ...]
          valid_mask: [B, T_max] float32 1 = real, 0 = pad
          lengths: [B] int — **effective** transitions in this batch row ``min(episode_length, T_max)``
            (matches copied data and ``valid_mask``; required for TD unroll / masking).
          state_frames, next_state_frames: optional [B, T_max, ...] or None
        """
        n = len(self.episodes)
        if n == 0:
            return None

        rng = rng or np.random.default_rng()
        idx = rng.choice(n, size=batch_size, replace=True)

        batch = [self.episodes[i] for i in idx]
        raw_lengths = np.array([ep['length'] for ep in batch], dtype=np.int32)
        if max_len is not None:
            T = int(min(max_len, int(raw_lengths.max())))
        else:
            T = int(raw_lengths.max())

        B = len(batch)
        D = batch[0]['states'].shape[1]
        n_actions = batch[0]['actions'].shape[1]

        states = np.zeros((B, T, D), dtype=np.float32)
        next_states = np.zeros((B, T, D), dtype=np.float32)
        actions = np.zeros((B, T, n_actions), dtype=np.float32)
        rewards = np.zeros((B, T), dtype=np.float32)
        dones = np.zeros((B, T), dtype=np.bool_)
        valid = np.zeros((B, T), dtype=np.float32)

        sf_out = None
        st_out = None
        nst_out = None
        if 'state_frames' in batch[0]:
            fh, fw, fc = batch[0]['state_frames'].shape[1:]
            sf_out = np.zeros((B, T, fh, fw, fc), dtype=np.float16)
            nsf_out = np.zeros((B, T, fh, fw, fc), dtype=np.float16)
        if 'state_times' in batch[0]:
            st_out = np.zeros((B, T), dtype=np.int32)
            nst_out = np.zeros((B, T), dtype=np.int32)

        lengths = np.zeros(B, dtype=np.int32)
        for b, ep in enumerate(batch):
            L = int(min(ep['length'], T))
            lengths[b] = L
            states[b, :L] = ep['states'][:L]
            next_states[b, :L] = ep['next_states'][:L]
            actions[b, :L] = ep['actions'][:L]
            rewards[b, :L] = ep['rewards'][:L]
            dones[b, :L] = ep['dones'][:L]
            valid[b, :L] = 1.0
            if sf_out is not None:
                sf_out[b, :L] = ep['state_frames'][:L]
                nsf_out[b, :L] = ep['next_state_frames'][:L]
            if st_out is not None:
                st_out[b, :L] = ep['state_times'][:L]
                nst_out[b, :L] = ep['next_state_times'][:L]

        out = {
            'states': states,
            'next_states': next_states,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'valid_mask': valid,
            'lengths': lengths,
        }
        if sf_out is not None:
            out['state_frames'] = sf_out
            out['next_state_frames'] = nsf_out
        if st_out is not None:
            out['state_times'] = st_out
            out['next_state_times'] = nst_out
        return out
