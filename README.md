# MineRL Agent

Exploratory work for the [NeurIPS 2019: MineRL Competition.](https://www.aicrowd.com/challenges/neurips-2019-minerl-competition)

The goal of this competition was to train a reinforcement learning agent to collect a diamond in Minecraft based on a few human demonstrations. Collecting a diamond in Minecraft is a challenging for players, as it requires understanding of the terrain and of the tiered requirements to craft the necessary items. This is an extremely challenging task that is currently beyond the reach for state of the art RL algorithms due to the long delays between rewards and the few human demonstrations.

The consensus at the start of the competition was that the most promising approach for this problem is deep hierarchical RL. Deep hierarchical RL suffers from the same problem that all deep neural networks suffer from: Sample inefficiency and an inability to model causal relations. My plan to remedy both issues was to build a two-tiered system where a casual inference model that understands the high level structure of the environment is combined with a low-level DQN model. The downside of this approach is that the agent is not end-to-end differentiable and that proved to be an insurmountable challenge.

I've built a sequence state encoder to provide an interface between the causal inference model and DQN. Unfortunately, even after much experimentation, this state encoder hasn't produced useful results. This is the curse of representation learning: unless the model is trained with a loss function closely related to the task, it will not perform well. This is of course not possible for the casual inference model, so I've opted for an approach with a high-fidelity reconstruction loss.

This repository contains a state encoding sequence model with a DCGAN loss. I've also experimented with  Wasserstein loss and simple regression loss on the inventory values. Neither of these models converged even after much fine-tuning.