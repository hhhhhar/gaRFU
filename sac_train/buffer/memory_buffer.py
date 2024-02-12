import random

class Replay_Buffer(object):
    def __init__(self, buffer_size, transition, batch_size):
        self.buffer_size = buffer_size
        self.transition = transition
        self.batch_size = batch_size
        self.store_number = 0
        self.memory = []

    def store_transition(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.buffer_size:
            self.memory.append(None)
        index = self.store_number % self.buffer_size
        self.memory[index] = self.transition(*args)
        self.store_number += 1

    def sample(self):
        transitions = random.sample(self.memory, self.batch_size)
        # batch = self.transition(*zip(*transitions))
        return transitions

    def sample_batch(self):
        transitions = random.sample(self.memory, self.batch_size)
        batch = self.transition(*zip(*transitions))
        return batch


