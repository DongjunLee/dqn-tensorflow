
class Config:

    def __init__(self, env):
        self.input_size = env.observation_space.shape[0]
        self.output_size = env.action_space.n
