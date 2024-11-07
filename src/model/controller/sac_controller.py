import logging
import os
import random

import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

from src.model.controller.replay_memory import ReplayMemory
from src.model.controller.base_controller import BaseController
from src.model.controller.policy import Policy


class SACController(BaseController):
    """
    SAC controller
    """
    def __init__(self, archspace, hyperspace, device, args, weight_directory='NTU_weights', **kwargs):
        super(SACController, self).__init__(**kwargs)

        self.args = args
        self.converged = False
        self.gpu_id = device

        # save as a string so logger can log
        self.reward_map_fn_str = 'lambda x: x'
        self.reward_map_fn = eval(self.reward_map_fn_str)

        # use average reward as baseline for rollouts
        self.use_baseline = self.args.use_baseline

        self.replay_memory = ReplayMemory(capacity=1000, threshold=)

        self.epsilon = 0.1

        # track policies for architecture space, hyperparameter space
        self.policies = {'archspace': {}, 'hpspace': {'optimizers': {}, 'lr': {}}}

        self.arch_space = archspace

        for idx, (key, value) in enumerate(self.arch_space.items()):
            n_comps = len(value)
            self.policies['archspace'][idx] = Policy(n_comps, self.gpu_id)

        # hyperparameter space
        optimizers = list(self.args.hyper.optimizers)
        learning_rates = list(self.args.hyper.lr)
        self.hpspace = hyperspace
        # these should probably be OrderedDicts to make life easier
        self.policies['hpspace']['optimizers'] = Policy(len(optimizers), self.gpu_id)
        self.policies['hpspace']['lr'] = Policy(len(learning_rates), self.gpu_id)

        # params for optimizer
        parameters = [self.policies['archspace'][i].parameters() for i in self.policies['archspace']]
        parameters += [self.policies['hpspace']['optimizers'].parameters()]
        parameters += [self.policies['hpspace']['lr'].parameters()]
        parameters = [{'params': p} for p in parameters]

        # optimizer for parameters
        self.optimizer_controller = torch.optim.Adam(parameters, lr=self.args.controller_lr)



        logging.info("Controller optimizer is Adam with lr {}".format(self.args.controller_lr))

    def has_converged(self) -> bool:
        """
        Track convergence.
        """
        return self.converged

    def sample(self, arch_choices, arch_computations):
        """
        Randomly sample the model parameters and set of hyperparameters from combined space
        """
        architecture_actions = []
        hp_actions = []

        for i in range(arch_computations):
            action = Categorical(self.policies['archspace'][i]()).sample()
            architecture_actions.append(action)

        optimizer = Categorical(self.policies['hpspace']['optimizers']()).sample()
        learning_rate = Categorical(self.policies['hpspace']['lr']()).sample()
        hp_actions.append([optimizer, learning_rate])
        return architecture_actions, hp_actions

    def policy_argmax(self, arch_computations):
        """
        Return most likely candidate model and hyperparameters from combined space
        """
        layerwise_actions = []
        for i in range(arch_computations):
            action = torch.argmax(self.policies['archspace'][i].params)
            layerwise_actions.append(action)

        optimizer = torch.argmax(self.policies['hpspace']['optimizers'].params)
        learning_rate = torch.argmax(self.policies['hpspace']['lr'].params)
        hp_actions = [optimizer, learning_rate]
        return layerwise_actions, hp_actions

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



    def update(self, rollouts):
        """
        Perform update step of REINFORCE
        args:
            rollouts: `n` long list like [(model_params, hp_params, quality), ...]
        """
        # TODO other algorithm?
        # in this case top1 accuracy
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

        """
        # calculate log probabilities for each time step
        log_prob = []
        for t in rollouts:
            _log_prob = []
            layerwise_actions, hp_actions_ = t[:2]
            for i in range(len(self.policies['archspace'])):
                layer_action, layer_policy = layerwise_actions[i], self.policies['archspace'][i]
                _log_prob.append(Categorical(layer_policy()).log_prob(layer_action))

            i = 0
            for key in self.policies['hpspace']:
                policy = self.policies['hpspace'][key]
                _log_prob.append(Categorical(policy()).log_prob(hp_actions_[0][i]))
                i += 1
            log_prob.append(torch.stack(_log_prob).sum())
        """
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

        """
        # old implementation
        loss = [-r * lp for r, lp in zip(rewards, log_prob)]
        loss = torch.stack(loss).sum()
        loss.backward()
        self.optimizer_controller.step()
        logging.info('')
        """

    def save_policies(self, directory):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        if directory[-1] != '/':
            directory += '/'
        for k in self.policies['archspace']:
            torch.save(self.policies['archspace'][k].state_dict(), directory + 'archspace_' + str(k))
        for k in self.policies['hpspace']:
            torch.save(self.policies['hpspace'][k].state_dict(), directory + 'hpspace_' + k)

    def load_policies(self, directory='ntu_controller_weights/'):
        if not os.path.isdir(directory):
            raise ValueError('Directory %s does not exist' % directory)

        if directory[-1] != '/':
            directory += '/'
        for k in self.policies['archspace']:
            _ = torch.load(directory + 'archspace_' + str(k))
            self.policies['archspace'][k].load_state_dict(_)
        for k in self.policies['hpspace']:
            _ = torch.load(directory + 'hpspace_' + k)
            self.policies['hpspace'][k].load_state_dict(_)
