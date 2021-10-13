import copy
import time
import numpy as np
from components.episode_buffer import ReplayBuffer


def launch(args, training_start_time, container_device, scheme, groups, preprocess, env_info,
           buffer_receiver_in_queues, buffer_receiver_out_queue, buffer_receiver_out_flag,
           container_id):
    buffer = ReplayBuffer(scheme, groups, args.container_buffer_receiver_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess, device=container_device)
    buffer_receiver_in_queue_i = 0
    buffer_receiver_in_queue_n = len(buffer_receiver_in_queues)

    ##################### log stuff #####################
    if args.log_container_buffer_receiver:
        log_last_time = time.time()
        log_frequency = 60
        log_n = 200
        log_i = 0
        log_max_episode_in_buffer = 0
        log_buffer_receiver_output_time = []
    #####################################################

    while time.time() - training_start_time < args.training_time:
        while buffer_receiver_out_flag < 2 or buffer.episodes_in_buffer == 0:
            if buffer_receiver_in_queues[buffer_receiver_in_queue_i].empty():
                buffer_receiver_in_queue_i = (buffer_receiver_in_queue_i + 1) % buffer_receiver_in_queue_n
            else:
                _batch = buffer_receiver_in_queues[buffer_receiver_in_queue_i].get()
                batch = copy.deepcopy(_batch)
                del _batch
                if batch.device != container_device:
                    batch.to(container_device)
                buffer.insert_episode_batch(batch)

        ##################### log stuff #####################
        if args.log_container_buffer_receiver:
            log_max_episode_in_buffer = max(log_max_episode_in_buffer, buffer.episodes_in_buffer)
            log_put_time = -time.time()
        #####################################################

        buffer_receiver_out_queue.put(copy.deepcopy(buffer.sample(buffer.episodes_in_buffer)))

        ##################### log stuff #####################
        if args.log_container_buffer_receiver:
            log_put_time += time.time()
            if len(log_buffer_receiver_output_time) < log_n:
                log_buffer_receiver_output_time.append(log_put_time)
            else:
                log_buffer_receiver_output_time[log_i] = log_put_time
                log_i = (log_i + 1) % log_n
            if time.time() - log_last_time > log_frequency:
                log_last_time = time.time()
                print("container id={} buffer receiver: max size {}, average output time {};".format(container_id, log_max_episode_in_buffer,
                                                                                              np.mean(log_buffer_receiver_output_time)))
        #####################################################

        buffer = ReplayBuffer(scheme, groups, args.container_buffer_receiver_size, env_info["episode_limit"] + 1,
                              preprocess=preprocess, device=container_device)
        buffer_receiver_out_flag -= 1
