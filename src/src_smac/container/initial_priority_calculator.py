import copy
import time
import numpy as np
import torch as th
import math
from learners.q_learner import PriorityCalculator


def launch(args, training_start_time, container_device,
           container_mac, container_target_mac, container_mixer, container_target_mixer,
           in_queue, container_out_queue, centralizer_out_queue,
           container_id):
    priority_calculator = PriorityCalculator(copy.deepcopy(container_mac), args)
    if container_device != "cpu":
        th.cuda.set_device(container_device)
        priority_calculator.cuda()

    ################### log stuff ####################
    if args.log_container_initial_priority_calculator:
        log_last_time = time.time()
        log_frequency = 60
        log_n = 200
        log_i = 0
        log_container_send_time = []
        log_centralizer_send_time = []
    ##################################################

    with th.no_grad():
        while time.time() - training_start_time < args.training_time:
            if not in_queue.empty():
                _batch = in_queue.get()
                batch = copy.deepcopy(_batch)
                del _batch
                if batch.device != container_device:
                    batch.to(container_device)
                priorities = priority_calculator.compute(batch, container_mac, container_target_mac, container_mixer, container_target_mixer)

                ################### log stuff ####################
                if args.log_container_initial_priority_calculator:
                    log_centralizer_time = -time.time()
                ##################################################

                ep_ids = np.random.choice(batch.batch_size, size=math.ceil(batch.batch_size * args.centralizer_receive_ratio),
                                          replace=False, p=priorities / priorities.sum())
                centralizer_out_queue.put(copy.deepcopy(batch[ep_ids]))

                ################### log stuff ####################
                if args.log_container_initial_priority_calculator:
                    log_centralizer_time += time.time()
                    log_container_time = -time.time()
                ##################################################

                container_out_queue.put(copy.deepcopy((batch, priorities)))

                ################### log stuff ####################
                if args.log_container_initial_priority_calculator:
                    log_container_time += time.time()
                    if len(log_container_send_time) < log_n:
                        log_container_send_time.append(log_container_time)
                        log_centralizer_send_time.append(log_centralizer_time)
                    else:
                        log_container_send_time[log_i] = log_container_time
                        log_centralizer_send_time[log_i] = log_centralizer_time
                        log_i = (log_i + 1) % log_n

                    if time.time() - log_last_time > log_frequency:
                        log_last_time = time.time()
                        print("container id={} initial priority calculator: average container send time {}, average centralizer send time {}, average reward {}, #compute = {};".format(
                            container_id, np.mean(log_container_send_time), np.mean(log_centralizer_send_time), np.mean(priority_calculator.log_avg_reward), priority_calculator.log_n_compute
                        ))
                ##################################################
