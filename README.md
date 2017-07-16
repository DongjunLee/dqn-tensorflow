# Deep Q Network
## Paper
- [playing atari with deep reinforcement learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) (NIPS 2013)
- [Human-Level Control through Deep Reinforcement Learning](https://www.nature.com/nature/journal/v518/n7540/full/nature14236.html) (NIPS 2015)


## TO DO

- Test: Atari
- model.json -> build_graph (MLP, ConvNet, RNN, etc..)

## Results

### Classic control

1. CartPole-v0 
	- defines "solving" as getting average reward of 195.0 over 100 consecutive trials.
	- **Model** : Shallow Network (hidden_size=128)
	-  Game Cleared in 155 episodes with avg reward 195.45

![images](images/CartPole-v0.gif)

2. CartPole-v1
	- defines "solving" as getting average reward of 475.0 over 100 consecutive trials.
	- **Model** : Shallow Network (hidden_size=128)
	- Game Cleared in 775 episodes with avg reward 477.2

![images](images/CartPole-v1.gif)

3. MountainCar-v0
	- defines "solving" as getting average reward of -110.0 over 100 consecutive trials.
	- **Model** : Shallow Network (hidden_size=128)
	- Game Cleared in 1586 episodes with avg reward -109.84

![images](images/MountainCar-v0.gif)



## Reference

- Base code : [humkim/ReinforcementZeroToAll](https://github.com/hunkim/ReinforcementZeroToAll/blob/master/07_3_dqn_2015_cartpole.py)
