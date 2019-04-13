### Report for Project 3 - Collaboration and Competition

## Learning Algorithm

The learning algorithm used was a Multi-Agent Deep Deterministic Policy Gradient (MADDPG) using two Actor-Critic models with underlying neural networks, all four of which used two fully connected layers, with 256 and 128 units for the first and second layer, respectively, and relu activation functions.

The hyperparameters used to train the agent were as follows:
- BUFFER_SIZE = int(1e6)  (replay buffer size)
- BATCH_SIZE = 128        (minibatch size)
- GAMMA = 0.99            (discount factor)
- TAU = 7e-2              (for soft update of target parameters)
- LR_ACTOR = 1e-3         (learning rate of the actor)
- LR_CRITIC = 1e-3        (learning rate of the critic)
- WEIGHT_DECAY = 0        (L2 weight decay)
- EPS_START = 5.5         (initial value for epsilon)
- EPS_EP_END = 250        (episode to end the noise decay process)
- EPS_FINAL = 0           (final value for epsilon after decay)
- OU_SIGMA = 0.2          (Ornstein-Uhlenbeck noise parameter, volatility)
- OU_THETA = 0.12         (Ornstein-Uhlenbeck noise parameter, speed of mean reversion)

## Plot of Rewards

The following is a plot of the rewards for the trained agent showing the first 2000 episodes. It took 704 to solve the problem, at which point the moving average score over the previous 100 episodes was 0.5:

![alt text](plot_of_rewards.png "Plot of Rewards")

## Ideas for Future Work

The initial implementation was able to solve the task, but did not maintain a very consistent, high score and the moving average dropped below the threshold multiple times after the initial solution was reached. Further optimization of hyperparameters could result in more consistent success, or even higher scores.


