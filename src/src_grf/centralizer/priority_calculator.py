import copy
import time
import numpy as np
import torch as th
from learners.q_learner import PriorityCalculator


def launch(args, device, training_start_time,
           priority_calculator_queue, priority_calculator_queue_size, priority_update_queue,
           mac, target_mac, mixer, target_mixer):
    priority_calculator = PriorityCalculator(copy.deepcopy(mac), args)
    if device != "cpu":
        th.cuda.set_device(device)
        priority_calculator.cuda()

    ################### log stuff ####################
    if args.log_centralizer_priority_calculator:
        log_last_time = time.time()
        log_frequency = 60
        log_n = 200
        log_i = 0
        log_wait_times = []
        log_wait_time = None
    ##################################################

    with th.no_grad():
        while time.time() - training_start_time < args.training_time:
            if not priority_calculator_queue.empty():
                ################### log stuff ####################
                if args.log_centralizer_priority_calculator:
                    if log_wait_time is not None:
                        log_wait_time += time.time()
                        if len(log_wait_times) < log_n:
                            log_wait_times.append(log_wait_time)
                        else:
                            log_wait_times[log_i] = log_wait_time
                            log_i = (log_i + 1) % log_n

                        if time.time() - log_last_time > log_frequency:
                            log_last_time = time.time()
                            print("centralizer priority calculator: average waiting time {}, average computing time {}, average length {}, #compute {}, average (priority: {}, reward: {}, covariance: {});".format(
                                np.mean(log_wait_times), np.mean(priority_calculator.log_computing_time),
                                np.mean(priority_calculator.log_avg_len), np.mean(priority_calculator.log_n_compute),
                                np.mean(priority_calculator.log_avg_priority), np.mean(priority_calculator.log_avg_reward), np.mean(priority_calculator.log_avg_cov)))
                ##################################################

                _ep_ids, _batch, _sample_time = priority_calculator_queue.get()
                ep_ids = copy.deepcopy(_ep_ids)
                batch = copy.deepcopy(_batch)
                sample_time = copy.deepcopy(_sample_time)
                del _ep_ids, _batch, _sample_time
                priority_calculator_queue_size -= 1

                if batch.device != device:
                    batch.to(device)

                priority = priority_calculator.compute(batch, mac, target_mac, mixer, target_mixer)
                priority_update_queue.put(copy.deepcopy((ep_ids, priority, sample_time)))

                ################### log stuff ####################
                if args.log_centralizer_priority_calculator:
                    log_wait_time = -time.time()
                ##################################################
