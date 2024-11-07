import logging
import os

import numpy as np
import torch
from torch.distributions import Categorical

from src.model.controller.replay_memory import ReplayMemory
from src.model.controller.base_controller import BaseController
from src.model.controller.policy import Policy


class PPOController(BaseController):

    def __init__(self, archspace, hyperspace, device, args, **kwargs):
        super(PPOController, self).__init__(**kwargs)
        pass

    def update(self, reward):
        pass

    def has_converged(self) -> bool:
        raise NotImplementedError

    def sample(self, choices, computations):
        raise NotImplementedError

    def policy_argmax(self, computations):
        raise NotImplementedError

    def update(self, reward_signal):
        raise NotImplementedError




@staticmethod
def ppo_loss(old_action_probs, new_action_probs, advantages, returns, clip_epsilon):
    ratio = torch.exp(torch.log(new_action_probs) - torch.log(old_action_probs))
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()
    critic_loss = F.smooth_l1_loss(returns, values)
    total_loss = actor_loss + critic_loss
    return total_loss

def ppo_update(self, policy, optimizer, states, actions, returns, advantages, clip_epsilon, num_epochs):
    for _ in range(num_epochs):
        action_probs, values = policy(states)
        new_action_probs = action_probs.gather(1, actions.unsqueeze(1))

        loss = self.ppo_loss(new_action_probs, action_probs, advantages, returns, clip_epsilon)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def compute_advantages(self, rewards, values, gamma, lam):
    advantages = []
    next_advantage = 0

    for reward, value in zip(reversed(rewards), reversed(values)):
        td_error = reward + gamma * value - value
        next_advantage = td_error + gamma * lam * next_advantage
        advantages.insert(0, next_advantage)

    returns = torch.Tensor(advantages) + torch.Tensor(values)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages

def update_ppo(self, rollouts):

    clip_epsilon = 0.2
    gamma = 0.99
    lam = 0.95

    rewards = [i[2]['acc_top1'] for i in rollouts]

    # delete best_acc of 0 --> model was not trained
    # rewards = [i for i in rewards if i != 0]

    # exponentiate rewards
    if self.reward_map_fn:
        rewards = [self.reward_map_fn(r) for r in rewards]

    # calculate rewards using average reward as baseline
    if self.use_baseline and len(rollouts) > 1:
        avg_reward = np.mean(rewards)
        rewards = [r - avg_reward for r in rewards]

    returns, advantages = self.compute_advantages(rewards, values.detach(), gamma, lam)

    self.optimizer_controller.zero_grad()

    for rollout in rollouts:
        layerwise_actions, hp_actions_ = rollout[:2]
        _log_prob = []
        for i in range(len(self.policies['archspace'])):
            layer_action, layer_policy = layerwise_actions[i], self.policies['archspace'][i]
            _log_prob.append(Categorical(layer_policy()).log_prob(layer_action))

        i = 0
        for key in self.policies['hpspace']:
            policy = self.policies['hpspace'][key]
            _log_prob.append(Categorical(policy()).log_prob(hp_actions_[0][i]))
            i += 1

        old_log_prob = torch.stack(_log_prob).sum().detach()
        reward = rollout[2]['acc_top1']

        new_log_prob = torch.stack(_log_prob).sum()
        ratio = torch.exp(new_log_prob - old_log_prob)
        surr1 = ratio * reward
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * reward

        loss = -torch.min(surr1, surr2)
        advantages = reward

        # Store the rollout in the replay memory
        self.replay_memory.push(rollout)

        # Sample from replay memory
        replay_batch = self.replay_memory.sample(batch_size=32)
        for replay_rollout in replay_batch:
            replay_reward = replay_rollout[2]['acc_top1']
            # Perform update using the replay reward
            # ... Your update step using replay_reward ...

        self.optimizer_controller.zero_grad()
        loss.backward()
        self.optimizer_controller.step()
