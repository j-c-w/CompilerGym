from compiler_gym.spaces import Reward

class RuntimeReward(Reward):
    """An example reward that uses changes in the "runtime" observation value
    to compute incremental reward.
    """

    def __init__(self):
        super().__init__(
            name="II",
            observation_spaces=["II"],
            default_value=0,
            default_negates_returns=True,
            deterministic=True,
            platform_dependent=True,
        )
        pass

    def reset(self, benchmark: str, observation_view):
        del benchmark  # unused

    def update(self, action, observations, observation_view):
        del action
        del observation_view

        print("Computing Reward: got II of ", observations[0])
        # Add a constant negative reward for not figuring it out?
        return -float(observations[0]) - 0.1


