import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop
import time


class QLearner:
    def __init__(self, mac, args, logger=None):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(self.mac.parameters())

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
        if self.args.log_centralizer_learner:
            self.log_n = 200
            self.log_i = 0
            self.log_priority = []
            self.log_loss = []
            self.log_training_time = []
            self.log_avg_len = []
        ######################################################

    def train(self, batch: EpisodeBatch, t_env: int):
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

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

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

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.cur_episode += 1
        if self.cur_episode - self.last_target_update_episode >= self.args.target_update_interval:
            self._update_targets()
            self.last_target_update_episode = self.cur_episode

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
        # res = th.where(th.sum(rewards.reshape(rewards.shape[0], -1), dim=-1) > 0, 0.5, 0.1)
        res = th.sum(rewards.reshape(rewards.shape[0], -1), dim=-1) / 21.0
        res = res.cpu().detach().numpy()

        ##################### log stuff #####################
        if self.args.log_centralizer_learner:
            log_training_time += time.time()
            if len(self.log_loss) < self.log_n:
                self.log_loss.append(loss.item())
                self.log_priority.append(float(res.mean()))
                self.log_training_time.append(log_training_time)
                self.log_avg_len.append(batch.max_seq_length)
            else:
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


class PriorityCalculator:
    def __init__(self, mac, args):
        self.args = args
        self.mac = mac

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.target_mixer = copy.deepcopy(self.mixer)

        self.target_mac = copy.deepcopy(mac)

        ################### log stuff ####################
        if args.log_priority_calculator:
            self.log_n_compute = 0
            self.log_n = 200
            self.log_i = 0
            self.log_computing_time = []
            self.log_avg_len = []
            self.log_avg_priority = []
            self.log_avg_reward = []
            self.log_avg_cov = []
        ##################################################

    def compute(self, batch: EpisodeBatch, mac, target_mac, mixer, target_mixer):
        with th.no_grad():
            ################### log stuff ####################
            if self.args.log_priority_calculator:
                self.log_n_compute += 1
                log_computing_time = -time.time()
            ##################################################

            self.mac.load_state(mac)
            self.target_mac.load_state(target_mac)
            self.mixer.load_state_dict(mixer.state_dict())
            self.target_mixer.load_state_dict(target_mixer.state_dict())

            # Get the relevant quantities
            rewards = batch["reward"][:, :-1]
            actions = batch["actions"][:, :-1]
            terminated = batch["terminated"][:, :-1].float()
            mask = batch["filled"][:, :-1].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
            avail_actions = batch["avail_actions"]

            # Calculate estimated Q-Values
            mac_out = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time

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

            # Calculate 1-step Q-Learning targets
            targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

            # Td-error
            td_error = (chosen_action_qvals - targets.detach())

            mask = mask.expand_as(td_error)

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask

            '''
            abs_td_error = th.abs(masked_td_error)
            assert abs_td_error.shape[2] == 1
            tde_mean = th.sum(abs_td_error, dim=(1, 2)) / th.sum(mask, dim=(1, 2))
            tde_max = th.max(abs_td_error.squeeze(2), dim=1).values
            res = self.args.priority_eta * tde_max + (1 - self.args.priority_eta) * tde_mean
            res = res.cpu().detach().numpy()
            '''
            # res = th.where(th.sum(rewards.reshape(rewards.shape[0], -1), dim=-1) > 0, 0.5, 0.1)
            res = th.sum(rewards.reshape(rewards.shape[0], -1), dim=-1) / 21.0
            res = res.cpu().detach().numpy()

            res_reward = th.sum(rewards.reshape(rewards.shape[0], -1), dim=-1) / 21.0
            res_reward = res_reward.cpu().detach().numpy()

            ################### log stuff ####################
            if self.args.log_priority_calculator:
                log_computing_time += time.time()
                if len(self.log_computing_time) < self.log_n:
                    self.log_computing_time.append(log_computing_time)
                    self.log_avg_len.append(batch.max_seq_length)
                    self.log_avg_priority.append(res.mean())
                    self.log_avg_reward.append(res_reward.mean())

                    cov_x = res - res.mean()
                    cov_y = res_reward - res_reward.mean()
                    cov = cov_x * cov_y
                    cov = cov.sum() / (batch.batch_size - 1)

                    self.log_avg_cov.append(cov)
                else:
                    self.log_computing_time[self.log_i] = log_computing_time
                    self.log_avg_len[self.log_i] = batch.max_seq_length
                    self.log_avg_priority[self.log_i] = res.mean()
                    self.log_avg_reward[self.log_i] = res_reward.mean()

                    cov_x = res - res.mean()
                    cov_y = res_reward - res_reward.mean()
                    cov = cov_x * cov_y
                    cov = cov.sum() / (batch.batch_size - 1)

                    self.log_avg_cov[self.log_i] = cov
                    self.log_i = (self.log_i + 1) % self.log_n
            ##################################################

            return res

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
