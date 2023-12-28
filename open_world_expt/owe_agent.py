
class OweAgent:
    def __init__(self):
        self.name = 'owe_agent'

    def action(self, observation, info, reward=None, terminated=None, truncated=None):
        raise NotImplementedError

    def novelty_detection(self):
        raise NotImplementedError


