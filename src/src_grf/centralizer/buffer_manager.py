import copy
import time
import numpy as np
from components.episode_buffer import PrioritizedReplayBuffer


def launch(args, training_start_time, device, scheme, groups, preprocess, env_info,
           buffer_receiver_out_flag, buffer_manager_in_queue,
           learner_device, learner_queue, learner_queue_size,
           priority_calculator_device, priority_calculator_queue, priority_calculator_queue_size,
           priority_update_queue):
    buffer = PrioritizedReplayBuffer(scheme, groups, args.centralizer_buffer_size, env_info["episode_limit"] + 1, args.alpha,
                                     preprocess=preprocess, device=device)

    ##################### log stuff #####################
    if args.log_centralizer_buffer_manager:
        log_last_time = time.time()
        log_frequency = 60
        log_buffer_size = 0
    #####################################################

    while time.time() - training_start_time < args.training_time:
        if not buffer_manager_in_queue.empty():
            buffer_receiver_out_flag -= 1
            _batch, _priority = buffer_manager_in_queue.get()
            batch = copy.deepcopy(_batch)
            if batch.device != device:
                batch.to(device)
            priority = copy.deepcopy(_priority)
            del _batch, _priority
            buffer.insert_episode_batch(batch, priority)

            ##################### log stuff #####################
            if args.log_centralizer_buffer_manager:
                log_buffer_size += batch.batch_size
                if time.time() - log_last_time > log_frequency:
                    log_last_time = time.time()
                    print("centralizer buffer: size {};".format(log_buffer_size))
            #####################################################

        if (buffer_receiver_out_flag < 1) and buffer_manager_in_queue.empty():
            buffer_receiver_out_flag += 2

        while not priority_update_queue.empty():
            _ep_ids, _priorities, _sample_time = priority_update_queue.get()
            ep_ids = copy.deepcopy(_ep_ids)
            priorities = copy.deepcopy(_priorities)
            sample_time = copy.deepcopy(_sample_time)
            del _ep_ids, _priorities, _sample_time
            buffer.update_priority(ep_ids, priorities, sample_time)

        if buffer.can_sample(args.centralizer_learner_batch_size):
            while int(learner_queue_size) < args.centralizer_learner_queue_size:
                ep_ids, batch, sample_time = buffer.proportional_sample(args.centralizer_learner_batch_size)
                max_ep_t = batch.max_t_filled()
                batch = batch[:, :max_ep_t]
                batch.to(learner_device)
                learner_queue.put(copy.deepcopy((ep_ids, batch, sample_time)))
                learner_queue_size += 1

        if buffer.can_sample(args.centralizer_priority_calculator_batch_size):
            while int(priority_calculator_queue_size) < args.centralizer_priority_calculator_queue_size:
                ep_ids, batch, sample_time = buffer.uniform_sample(args.centralizer_priority_calculator_batch_size)
                max_ep_t = batch.max_t_filled()
                batch = batch[:, :max_ep_t]
                batch.to(priority_calculator_device)
                priority_calculator_queue.put(copy.deepcopy((ep_ids, batch, sample_time)))
                priority_calculator_queue_size += 1

    buffer_receiver_out_flag += 2
