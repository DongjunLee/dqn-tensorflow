
class Config:

    def __init__(self, env, gym_env: str) -> None:
        self.input_size = env.observation_space.shape[0]
        self.output_size = env.action_space.n

        self.solving_criteria = None
        if gym_env == "CartPole-v0":
            self.solving_criteria = (100, 195)
        elif gym_env == "CartPole-v1":
            self.solving_criteria = (100, 475)
        elif gym_env == "MountainCar-v0":
            self.solving_criteria = (100, -110)
