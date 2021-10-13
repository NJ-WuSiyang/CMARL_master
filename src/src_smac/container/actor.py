from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import copy
import time
import torch as th


class EpisodeRunner:

    def __init__(self, args):
        self.args = args
        self.batch_size = self.args.actor_buffer_size
        self.trajectories_in_batch = self.batch_size - 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        ################### log stuff ####################
        if args.log_actor:
            self.log_last_time = time.time()
            self.log_frequency = 60
            self.log_n = 200
            self.log_i = 0
            self.log_send_time = []
        ##################################################

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.trajectories_in_batch += 1
        if self.trajectories_in_batch == self.batch_size:
            self.batch = self.new_batch()
            self.trajectories_in_batch = 0

        self.env.reset()
        self.t = 0

    def run(self, container_experience_receiver, test_mode=False, container_id=None):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=1)

        epsilon_base = self.mac.action_selector.schedule.eval(self.t_env)
        epsilon = np.random.uniform(epsilon_base * self.args.epsilon_lb, min(1., epsilon_base * self.args.epsilon_rb))

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t, bs=self.trajectories_in_batch)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch[self.trajectories_in_batch], t_ep=self.t, t_env=self.t_env, test_mode=test_mode, epsilon=epsilon)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t, bs=self.trajectories_in_batch)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t, bs=self.trajectories_in_batch)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch[self.trajectories_in_batch], t_ep=self.t, t_env=self.t_env, test_mode=test_mode, epsilon=epsilon)
        self.batch.update({"actions": actions}, ts=self.t, bs=self.trajectories_in_batch)

        if self.trajectories_in_batch == self.batch_size - 1:
            ##################### log stuff #####################
            if self.args.log_actor:
                log_put_time = -time.time()
            #####################################################

            # TODO: try the different version: first transfer batch to gpu, then send the batch
            # TODO: transmit batch with time dimension to max_T to reduce data transfer cost
            container_experience_receiver.put(copy.deepcopy(self.batch))

            ##################### log stuff #####################
            if self.args.log_actor:
                log_put_time += time.time()
                if len(self.log_send_time) < self.log_n:
                    self.log_send_time.append(log_put_time)
                else:
                    self.log_send_time[self.log_i] = log_put_time
                    self.log_i = (self.log_i + 1) % self.log_n
                if time.time() - self.log_last_time > self.log_frequency:
                    self.log_last_time = time.time()
                    print("container id={} actor: average sending time {}".format(container_id, np.mean(self.log_send_time)))
            #####################################################
        return self.t


def launch(_args, container_device, scheme, groups, preprocess,
           training_start_time, t_env,
           container_mac, centralizer_mac,
           container_experience_receiver,
           container_id):
    args = copy.deepcopy(_args)
    args.device = container_device
    if container_device == "cpu":
        args.use_cuda = False
    else:
        args.use_cuda = True
        th.cuda.set_device(container_device)

    runner = EpisodeRunner(args)
    mac = copy.deepcopy(container_mac)
    if args.use_cuda:
        mac.cuda()
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    ##################### log stuff #####################
    if args.log_actor:
        log_n_trajectories = 0
        log_last_time = time.time()
        log_frequency = 60
    #####################################################

    with th.no_grad():
        while time.time() - training_start_time < args.training_time:
            if args.no_exploration_diversity:
                runner.mac.load_state(centralizer_mac)
            else:
                runner.mac.load_state(container_mac)

            runner.t_env = int(t_env)
            t = runner.run(container_experience_receiver, test_mode=False, container_id=container_id)
            t_env += t

            ##################### log stuff #####################
            if args.log_actor:
                log_n_trajectories += 1
                if time.time() - log_last_time > log_frequency:
                    log_last_time = time.time()
                    print("container id={} actor: collect {} trajectories;".format(container_id, log_n_trajectories))
            #####################################################

    runner.close_env()
