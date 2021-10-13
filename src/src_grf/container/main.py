import time
import copy
import torch as th
from learners.diverse_q_learner import DiverseQLearner
from container.actor import launch as launch_actor
from container.buffer_receiver import launch as launch_buffer_receiver
from container.initial_priority_calculator import launch as launch_initial_priority_calculator
from container.buffer_manager import launch as launch_buffer_manager
from container.learner import launch as launch_learner
from container.priority_calculator import launch as launch_priority_calculator


def launch(ctx, args, container_id, scheme, groups, preprocess, env_info,
           learner_device, priority_calculator_device, buffer_device, actor_devices,
           centralizer_receiver_queue,
           training_start_time, t_env,
           centralizer_mac, container_mac, container_qs,
           centralizer_mac_time_step):
    _learner = DiverseQLearner(copy.deepcopy(container_mac), args)
    if learner_device != "cpu":
        th.cuda.set_device(learner_device)
        _learner.cuda()
    container_target_mac = _learner.target_mac
    container_mixer = _learner.mixer
    container_target_mixer = _learner.target_mixer
    container_mac.share_memory()
    container_target_mac.share_memory()
    container_mixer.share_memory()
    container_target_mixer.share_memory()

    priority_update_queue = ctx.SimpleQueue()

    learner_batch_queue = ctx.SimpleQueue()
    learner_batch_queue_size = th.tensor(0)
    learner_batch_queue_size.share_memory_()
    learner_p = ctx.Process(
        target=launch_learner,
        args=(args, learner_device, training_start_time, t_env, _learner,
              learner_batch_queue, learner_batch_queue_size, priority_update_queue,
              container_id, container_qs, centralizer_mac, container_mac, container_target_mac, container_mixer, container_target_mixer,
              centralizer_mac_time_step)
    )
    learner_p.start()

    priority_calculator_batch_queue = ctx.SimpleQueue()
    priority_calculator_batch_queue_size = th.tensor(0)
    priority_calculator_batch_queue_size.share_memory_()
    priority_calculator_p = ctx.Process(
        target=launch_priority_calculator,
        args=(args, priority_calculator_device, training_start_time,
              priority_calculator_batch_queue, priority_calculator_batch_queue_size, priority_update_queue,
              container_id, container_mac, container_target_mac, container_mixer, container_target_mixer)
    )
    priority_calculator_p.start()

    buffer_receiver_flag = th.tensor(0)
    buffer_receiver_flag.share_memory_()
    buffer_manager_in_queue = ctx.SimpleQueue()
    buffer_manager_p = ctx.Process(
        target=launch_buffer_manager,
        args=(args, training_start_time, buffer_device, scheme, groups, preprocess, env_info,
              buffer_receiver_flag, buffer_manager_in_queue,
              learner_batch_queue, learner_batch_queue_size,
              priority_calculator_batch_queue, priority_calculator_batch_queue_size,
              priority_update_queue,
              container_id)
    )
    buffer_manager_p.start()

    buffer_receiver_out_queue = ctx.SimpleQueue()
    initial_priority_calculator_p = ctx.Process(
        target=launch_initial_priority_calculator,
        args=(args, training_start_time, buffer_device,
              container_mac, container_target_mac, container_mixer, container_target_mixer,
              buffer_receiver_out_queue, buffer_manager_in_queue, centralizer_receiver_queue,
              container_id)
    )
    initial_priority_calculator_p.start()

    buffer_receiver_in_queues = []
    for _ in range(args.n_container_buffer_receiver_queues):
        buffer_receiver_in_queues.append(ctx.SimpleQueue())
    buffer_receiver_p = ctx.Process(
        target=launch_buffer_receiver,
        args=(args, training_start_time, buffer_device, scheme, groups, preprocess, env_info,
              buffer_receiver_in_queues, buffer_receiver_out_queue, buffer_receiver_flag,
              container_id)
    )
    buffer_receiver_p.start()

    actor_ps = []
    buffer_receiver_in_queue_i = 0
    buffer_receiver_in_queue_n = len(buffer_receiver_in_queues)
    for actor_device in actor_devices:
        actor_p = ctx.Process(
            target=launch_actor,
            args=(args, actor_device, scheme, groups, preprocess,
                  training_start_time, t_env,
                  container_mac, centralizer_mac,
                  buffer_receiver_in_queues[buffer_receiver_in_queue_i],
                  container_id)
        )
        actor_p.start()
        actor_ps.append(actor_p)
        buffer_receiver_in_queue_i = (buffer_receiver_in_queue_i + 1) % buffer_receiver_in_queue_n

    while time.time() - training_start_time < args.training_time:
        continue

    for actor_p in actor_ps:
        actor_p.join()
    buffer_receiver_p.join()
    initial_priority_calculator_p.join()
    buffer_manager_p.join()
    priority_calculator_p.join()
    learner_p.join()
