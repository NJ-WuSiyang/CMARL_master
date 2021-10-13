import copy
import time
import numpy as np
import torch as th
from learners.q_learner import PriorityCalculator


def launch(args, training_start_time, device,
           mac, target_mac, mixer, target_mixer,
           in_queue, out_queue):
    priority_calculator = PriorityCalculator(copy.deepcopy(mac), args)
    if device != "cpu":
        th.cuda.set_device(device)
        priority_calculator.cuda()

    ################### log stuff ####################
    if args.log_centralizer_initial_priority_calculator:
        log_last_time = time.time()
        log_frequency = 60
        log_n = 200
        log_i = 0
        log_send_time = []
    ##################################################

    with th.no_grad():
        while time.time() - training_start_time < args.training_time:
            if not in_queue.empty():
                _batch = in_queue.get()
                batch = copy.deepcopy(_batch)
                del _batch
                if batch.device != device:
                    batch.to(device)
                priorities = priority_calculator.compute(batch, mac, target_mac, mixer, target_mixer)

                ################### log stuff ####################
                if args.log_centralizer_initial_priority_calculator:
                    log_time = -time.time()
                ##################################################

                out_queue.put(copy.deepcopy((batch, priorities)))

                ################### log stuff ####################
                if args.log_centralizer_initial_priority_calculator:
                    log_time += time.time()
                    if len(log_send_time) < log_n:
                        log_send_time.append(log_time)
                    else:
                        log_send_time[log_i] = log_time
                        log_i = (log_i + 1) % log_n

                    if time.time() - log_last_time > log_frequency:
                        log_last_time = time.time()
                        print("centralizer initial priority calculator: average send time {}, average reward {}, #compute = {};".format(
                            np.mean(log_send_time), np.mean(priority_calculator.log_avg_reward), priority_calculator.log_n_compute
                        ))
                ##################################################
