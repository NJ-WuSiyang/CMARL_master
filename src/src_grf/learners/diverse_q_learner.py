import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop
import time
import numpy as np


class DiverseQLearner:
    def __init__(self, mac, args, logger=None):
        self.centralizer_mac_time_step = -1

        self.args = args
        self.mac = mac
        self.logger = logger

        self.min_return = self.args.env_args["min_return"]
        self.max_return = self.args.env_args["max_return"]

        for param in self.mac.agent.fc1.parameters():
            param.requires_grad = False
        for param in self.mac.agent.rnn.parameters():
            param.requires_grad = False
        self.params = list(self.mac.agent.fc2.parameters())

        self.last_target_update_episode = 0
        self.cur_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
            for param in self.target_mixer.parameters():
                param.requires_grad = False

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        for param in self.target_mac.agent.fc1.parameters():
            param.requires_grad = False
        for param in self.target_mac.agent.rnn.parameters():
            param.requires_grad = False
        for param in self.target_mac.agent.fc2.parameters():
            param.requires_grad = False

        self.log_stats_t = -self.args.learner_log_interval - 1

        ##################### log stuff #####################
        if self.args.log_container_learner:
            self.log_n = 200
            self.log_i = 0
            self.log_pi_kl = []
            self.log_priority = []
            self.log_loss = []
            self.log_training_time = []
            self.log_avg_len = []
        ######################################################

    def train(self, batch: EpisodeBatch, t_env: int, global_mac, _container_qs, centralizer_mac_time_step):
        ##################### log stuff #####################
        if self.args.log_centralizer_learner:
            log_training_time = -time.time()
        ######################################################

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        self.mac.load_rnn_state(global_mac)
        if not (th.equal(self.mac.agent.fc1.weight, self.target_mac.agent.fc1.weight) and
                th.equal(self.mac.agent.fc1.bias, self.target_mac.agent.fc1.bias) and
                th.equal(self.mac.agent.rnn.weight_ih, self.target_mac.agent.rnn.weight_ih) and
                th.equal(self.mac.agent.rnn.weight_hh, self.target_mac.agent.rnn.weight_hh) and
                th.equal(self.mac.agent.rnn.bias_ih, self.target_mac.agent.rnn.bias_ih) and
                th.equal(self.mac.agent.rnn.bias_hh, self.target_mac.agent.rnn.bias_hh)):
            self._update_targets()
            print("CONTAINER UPDATE TARGETS")
        container_qs = copy.deepcopy(_container_qs).to(rewards.device)

        # Calculate estimated Q-Values
        mac_out = []
        diverse_mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, hidden_states = self.mac.forward(batch, t=t, extract_feature=True)
            mac_out.append(agent_outs)
            with th.no_grad():
                diverse_outs = container_qs(hidden_states)
                diverse_mac_out.append(diverse_outs.reshape(-1, self.args.n_agents, diverse_outs.shape[-2], diverse_outs.shape[-1]))
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        diverse_mac_out = th.stack(diverse_mac_out, dim=1)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        pi = th.softmax(self.args.diverse_beta1 * mac_out, dim=-1)
        pi_avg = th.mean(th.softmax(diverse_mac_out, dim=-1), dim=-2)
        pi_kl = th.sum(pi * th.log(pi / pi_avg), dim=-1)
        pi_kl = pi_kl.mean(dim=-1, keepdim=True)
        avg_pi_kl = pi_kl.sum() / mask.sum()

        def _intrinsic_factor():
            if t_env > self.args.diverse_start_anneal_time:
                # TODO: optimize this term
                return max(1 - self.args.diverse_anneal_rate * (t_env - self.args.diverse_start_anneal_time) / 1000000, 0)
            else:
                return 1

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum() + \
               _intrinsic_factor() * self.args.diverse_beta2 * ((avg_pi_kl - self.args.diverse_loss_target) ** 2)

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.cur_episode += 1

        if self.logger is not None:
            if t_env - self.log_stats_t >= self.args.learner_log_interval:
                self.logger.log_stat("loss", loss.item(), t_env)
                self.logger.log_stat("grad_norm", grad_norm, t_env)
                mask_elems = mask.sum().item()
                self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
                self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
                self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
                self.log_stats_t = t_env

        '''
        abs_td_error = th.abs(masked_td_error)
        assert abs_td_error.shape[2] == 1
        tde_mean = th.sum(abs_td_error, dim=(1, 2)) / th.sum(mask, dim=(1, 2))
        tde_max = th.max(abs_td_error.squeeze(2), dim=1).values
        res = self.args.priority_eta * tde_max + (1 - self.args.priority_eta) * tde_mean
        res = res.cpu().detach().numpy()
        '''
        res = (th.sum(rewards.reshape(rewards.shape[0], -1), dim=-1) - self.min_return) / (self.max_return - self.min_return) + 0.1
        res = res.cpu().detach().numpy()

        ##################### log stuff #####################
        if self.args.log_container_learner:
            log_training_time += time.time()
            if len(self.log_pi_kl) < self.log_n:
                self.log_pi_kl.append(avg_pi_kl.item())
                self.log_loss.append(loss.item())
                self.log_priority.append(float(res.mean()))
                self.log_training_time.append(log_training_time)
                self.log_avg_len.append(batch.max_seq_length)
            else:
                self.log_pi_kl[self.log_i] = avg_pi_kl.item()
                self.log_loss[self.log_i] = loss.item()
                self.log_priority[self.log_i] = float(res.mean())
                self.log_training_time[self.log_i] = log_training_time
                self.log_avg_len[self.log_i] = batch.max_seq_length
                self.log_i = (self.log_i + 1) % self.log_n
        ######################################################

        return res

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.logger is not None:
            self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
