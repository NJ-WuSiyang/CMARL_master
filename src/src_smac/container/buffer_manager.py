import copy
import time
import numpy as np
from components.episode_buffer import PrioritizedReplayBuffer


def launch(args, training_start_time, container_device, scheme, groups, preprocess, env_info,
           buffer_receiver_out_flag, buffer_manager_in_queue,
           learner_queue, learner_queue_size,
           priority_calculator_queue, priority_calculator_queue_size,
           priority_update_queue,
           container_id):
    buffer = PrioritizedReplayBuffer(scheme, groups, args.container_buffer_size, env_info["episode_limit"] + 1, args.alpha,
                                     preprocess=preprocess, device=container_device)

    while time.time() - training_start_time < args.training_time:
        if not buffer_manager_in_queue.empty():
            buffer_receiver_out_flag -= 1
            _batch, _priority = buffer_manager_in_queue.get()
            batch = copy.deepcopy(_batch)
            if batch.device != container_device:
                batch.to(container_device)
            priority = copy.deepcopy(_priority)
            del _batch, _priority
            buffer.insert_episode_batch(batch, priority)

        if (buffer_receiver_out_flag < 1) and buffer_manager_in_queue.empty():
            buffer_receiver_out_flag += 2

        while not priority_update_queue.empty():
            _ep_ids, _priorities, _sample_time = priority_update_queue.get()
            ep_ids = copy.deepcopy(_ep_ids)
            priorities = copy.deepcopy(_priorities)
            sample_time = copy.deepcopy(_sample_time)
            del _ep_ids, _priorities, _sample_time
            buffer.update_priority(ep_ids, priorities, sample_time)

        if buffer.can_sample(args.container_learner_batch_size):
            while int(learner_queue_size) < args.container_learner_queue_size:
                ep_ids, batch, sample_time = buffer.proportional_sample(args.container_learner_batch_size)
                max_ep_t = batch.max_t_filled()
                batch = batch[:, :max_ep_t]
                learner_queue.put(copy.deepcopy((ep_ids, batch, sample_time)))
                learner_queue_size += 1

        if buffer.can_sample(args.container_priority_calculator_batch_size):
            while int(priority_calculator_queue_size) < args.container_priority_calculator_queue_size:
                ep_ids, batch, sample_time = buffer.uniform_sample(args.container_priority_calculator_batch_size)
                max_ep_t = batch.max_t_filled()
                batch = batch[:, :max_ep_t]
                priority_calculator_queue.put(copy.deepcopy((ep_ids, batch, sample_time)))
                priority_calculator_queue_size += 1

    buffer_receiver_out_flag += 2
