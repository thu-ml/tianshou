# MCTS

This is an implementation for Monte Carlo Tree Search in various Reinforcement Learning applications.

## MCTS with deterministic environments

The agent is interacting with a deterministic environment. That is to say, next state and reward for a state-action pair is deterministic. 

### Node

Action nodes are not needed here since every state-action pair only lead to one state.

Elements for a node:

+ state: the state for current node
+ parent node: the parent node of this node on the tree
+ children: the next states the agent can reach by choosing an action from this node
+ prior: some external information for this node (default to be uniform)
> If an action is chosen, the next state and reward are gotten. I assume that the action and correspond reward are maintained in element children.

Optional elements (for UCT or Thompson Sampling):

+ W: the sum of all sampled values collected by this node (for UCT)
+ N: the number of times the node is sampled (for UCT)
+ Q: the estimation for the value of this node, i.e. W/N (for UCT)
+ U: upper bound value for this node (for UCT)
+ alpha, beta: parameters for the posterior distribution of the value of this node (for Thompson Sampling, beta distribution)
+ mu, sigma: parameters for the posterior distribution of the value of this node (for Thompson Sampling, Gaussian distribution)

### Selection
In the selection part, an action is chosen for the current node. 

If the action has been chosen before, then go to the corresponding child node and go on selection.

If not, stop selection and start expansion.

### Expansion

Send the state-action pair to the simulator and then the next state and a reward are returned by the simulator. Then initialize a new node with the next state. Add the action, reward and next state link to children. The prior information may be given for initialization.

Then go to rollout.

### Rollout

At the new leaf node, use a quick policy to play the game to some terminal state and return the collected reward along the trajectory to the leaf node. Use this collected reward to initialize the value of this node.

Another way is to send this state to some external estimator and use the returned result to initialize the value of this node.

Then turn to backpropagation to send this value backup.

### Backpropagation

From the leaf node to the root node, update all nodes that have been passed in this iteration.

For each node, a value is returned by its child node. Then add this value (might be multiplied by a discouting factor gamma) with the stored reward for this child to get a new value. The new value is used to update this node.

For UCT methods, the new value is add to W. Then add 1 to N. Q and U can be calculated out.

For Thompson Sampling, the new value is treated as a sample to update the posterior distribution.

Then return teh new value to its parent node.

Stop backpropagation until root node is reached. Then start selection again.

## MCTS with random environments

The agent is interacting with a random environment. That is to say, next state and reward for a state-action pair is not deterministic. We do not know the hidden dynamics and reward distribution. We can just get samples from the simulator.

### Node

Both state nodes and action nodes are needed here.

#### State nodes

Elements for a state node:

+ state: the state for current node
+ parent node: the parent action node of this node on the tree
+ children: the actions nodes chosen from this node
+ prior: some external information for this node (default to be uniform)

Optional elements (for UCT or Thompson Sampling):

+ V: the estimation for the value of this node (for UCT)
+ N: the number of times the node is sampled (for UCT)

#### Action nodes

Elements for a state node:

+ action: the action for current node
+ parent node: the parent state node of this node on the tree
+ expected_reward: the expected one-step reward when this action is chosen from the parent node
+ children: the states node sampled by this action

Optional elements (for UCT or Thompson Sampling):

+ W: the sum of all sampled values collected by this node (for UCT)
+ N: the number of times the node is sampled (for UCT)
+ Q: the estimation for the value of this node, i.e. W/N (for UCT)
+ U: upper bound value for this node (for UCT)
+ alpha, beta: parameters for the posterior distribution of the value of this node (for Thompson Sampling, beta distribution)
+ mu, sigma: parameters for the posterior distribution of the value of this node (for Thompson Sampling, Gaussian distribution)

### Selection
In the selection part, an action is chosen for the current state node. Then the state-action pair to the simulator and then the next state and a reward are returned by the simulator.

If the next state has been seen from this action node before, then go to the corresponding child node and go on selection.

If not, stop selection and start expansion.

### Expansion

Initialize a new node with the next state. Add this node to children of the parent action node. Then generate all possible children for this node. Initialize them.

The prior information may be given for initialization. 

Then go to rollout.

### Rollout

At the new state node, choose one action as a leaf node. Then use a quick policy to play the game to some terminal state and return the collected reward along the trajectory to the leaf node. Use this collected reward to initialize the Q value of this action node.

> TODO: if external estimated values are used here, is it proper to estimate Q or V?

Then turn to backpropagation to send this value backup.

### Backpropagation

From the leaf node to the root node, update all nodes that have been passed in this iteration.

#### For action nodes

For each action node, a V value is returned by its child state node. Then add this value (might be multiplied by a discouting factor gamma) with the expected_reward for this action node to get a new value. The new value is used to update this node.

For UCT methods, the new value is add to W. Then add 1 to N. Q and U can be calculated out.

For Thompson Sampling, the new value is treated as a sample to update the posterior distribution.

Then return the new value to its parent state node. 

#### For state nodes

For a state node, a Q value is returned by its child action node.

For UCT, calculate the new averaged V with Q and N. Then N pluses 1.

For Thompson sampling, just return the Q value to its parent action value as a sample.


Stop backpropagation until root node is reached. Then start selection again.

## MCTS for POMDPs

The agent is interacting under a partially observed environment. That is to say, we can only see observations and choose actions. A simulator is needed here. Every time I sent a state action pair to the simulator, it can return me a new state, a observation and a reward. There need a prior distribution of states for the root node.

### Node

We use observation nodes and action nodes here.

#### Observation nodes

Elements for an observation node:

+ observation: the observation for current node
+ parent node: the parent action node of this node on the tree
+ children: the actions nodes chosen from this node
+ prior: some external information for this node (default to be uniform)

Optional elements (for UCT or Thompson Sampling):

+ h: history information for this node
+ V: the estimation for the value of this node (for UCT)
+ N: the number of times the node is sampled (for UCT)

#### Action nodes

Elements for a state node:

+ action: the action for current node
+ parent node: the parent observation node of this node on the tree
+ expected_reward: the expected one-step reward when this action is chosen from the parent node
+ children: the observations node sampled by this action

Optional elements (for UCT or Thompson Sampling):

+ W: the sum of all sampled values collected by this node (for UCT)
+ N: the number of times the node is sampled (for UCT)
+ Q: the estimation for the value of this node, i.e. W/N (for UCT)
+ U: upper bound value for this node (for UCT)
+ alpha, beta: parameters for the posterior distribution of the value of this node (for Thompson Sampling, beta distribution)
+ mu, sigma: parameters for the posterior distribution of the value of this node (for Thompson Sampling, Gaussian distribution)

### Selection
In the selection part, we first need to sample a state from the root node's prior distribution.

At each observation node, an action *a* is chosen. Here we have a state *s* for this observation. Then we send the *(s,a)*  pair to the simulator and then a new state *s'* and a reward are returned by the simulator.

If the next state has been seen from this action node before, then go to the corresponding child node with the new state *s'* and go on selection.

If not, stop selection and start expansion.

### Expansion

Initialize a new observation node. Add this node to children of its parent action node. Then generate all possible children for this node. Initialize them.

The prior information may be given for initialization. 

Then go to rollout.

### Rollout

At the new observation node, choose one action as a leaf node. Then use a quick policy to play the game to some terminal state and return the collected reward along the trajectory to the leaf node. Use this collected reward to initialize the Q value of this action node.

> TODO: if external estimated values are used here, is it proper to estimate Q or V?

Then turn to backpropagation to send this value backup.

### Backpropagation

From the leaf node to the root node, update all nodes that have been passed in this iteration.

#### For action nodes

For each action node, a V value is returned by its child observation node. Then add this value (might be multiplied by a discouting factor gamma) with the expected_reward for this action node to get a new value. The new value is used to update this node.

For UCT methods, the new value is add to W. Then add 1 to N. Q and U can be calculated out.

For Thompson Sampling, the new value is treated as a sample to update the posterior distribution.

Then return the new value to its parent observation node. 

#### For observation nodes

For an observation node, a Q value is returned by its child action node.

For UCT, calculate the new averaged V with Q and N. Then N pluses 1.

For Thompson sampling, just return the Q value to its parent action value as a sample.


Stop backpropagation until root node is reached. Then start selection again.