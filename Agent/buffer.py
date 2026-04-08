import numpy as np


class ReplayBuffer():
    def __init__(self, max_size, state_dim, n_actions, alpha=0.7, frame_shape=None):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.alpha = alpha
        self.state_dim = state_dim
        self.frame_shape = frame_shape

        self.state_memory = np.zeros((self.mem_size, state_dim))
        self.new_state_memory = np.zeros((self.mem_size, state_dim))

        if frame_shape is not None:
            self.state_frames_memory = np.zeros((max_size,) + frame_shape, dtype=np.float16)
            self.next_state_frames_memory = np.zeros((max_size,) + frame_shape, dtype=np.float16)
        else:
            self.state_frames_memory = None
            self.next_state_frames_memory = None

        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.time = np.zeros(self.mem_size, dtype=np.int32)

        self.priority_memory = np.zeros(self.mem_size)

    def store_transition(self, state, action, reward, state_, done, t,
                         state_frames=None, next_state_frames=None):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.time[index] = t

        if self.state_frames_memory is not None:
            self.state_frames_memory[index] = state_frames.astype(np.float16)
            self.next_state_frames_memory[index] = next_state_frames.astype(np.float16)

        self.priority_memory[index] = self.priority_memory.max() if self.mem_cntr > 0 else 1.0

        self.mem_cntr += 1

    def get_priority(self, td_error):
        return (np.abs(td_error) + 1e-5) ** self.alpha

    def sample_buffer(self, batch_size, beta=0.2):
        max_mem = min(self.mem_cntr, self.mem_size)

        priorities = self.priority_memory[:max_mem]
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(max_mem, batch_size, p=probabilities)

        total_prob = len(self.priority_memory) * probabilities[indices]
        weights = (total_prob ** -beta) / max(total_prob ** -beta)

        states = self.state_memory[indices]
        actions = self.action_memory[indices]
        rewards = self.reward_memory[indices]
        states_ = self.new_state_memory[indices]
        dones = self.terminal_memory[indices]
        ts = self.time[indices]

        if self.state_frames_memory is not None:
            state_frames = self.state_frames_memory[indices]
            next_state_frames = self.next_state_frames_memory[indices]
            return (states, actions, rewards, states_, dones, ts, indices, weights,
                    state_frames, next_state_frames)

        return states, actions, rewards, states_, dones, ts, indices, weights

    def update_priorities(self, indices, errors):
        errors = np.squeeze(errors)
        for i, error in zip(indices, errors):
            self.priority_memory[i] = self.get_priority(error)
