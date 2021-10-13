import time
import copy
import torch as th
from learners.q_learner import QLearner
from centralizer.buffer_receiver import launch as launch_buffer_receiver
from centralizer.initial_priority_calculator import launch as launch_initial_priority_calculator
from centralizer.buffer_manager import launch as launch_buffer_manager
from centralizer.learner import launch as launch_learner
from centralizer.priority_calculator import launch as launch_priority_calculator


def launch(ctx, args, scheme, groups, preprocess, env_info,
           learner_device, priority_calculator_device, buffer_device,
           centralizer_receiver_queue,
           training_start_time, t_env,
           global_centralizer_mac, global_centralizer_mac_time_step):
    centralizer_mac = copy.deepcopy(global_centralizer_mac)
    centralizer_mac.share_memory()
    _learner = QLearner(copy.deepcopy(centralizer_mac), args)
    if learner_device != "cpu":
        th.cuda.set_device(learner_device)
        _learner.cuda()
    target_mac = _learner.target_mac
    mixer = _learner.mixer
    target_mixer = _learner.target_mixer
    target_mac.share_memory()
    mixer.share_memory()
    target_mixer.share_memory()

    priority_update_queue = ctx.SimpleQueue()

    learner_batch_queue = ctx.SimpleQueue()
    learner_batch_queue_size = th.tensor(0)
    learner_batch_queue_size.share_memory_()
    learner_p = ctx.Process(
        target=launch_learner,
        args=(args, learner_device, training_start_time, t_env, _learner,
              learner_batch_queue, learner_batch_queue_size, priority_update_queue,
              centralizer_mac, target_mac, mixer, target_mixer,
              global_centralizer_mac, global_centralizer_mac_time_step)
    )
    learner_p.start()

    priority_calculator_batch_queue = ctx.SimpleQueue()
    priority_calculator_batch_queue_size = th.tensor(0)
    priority_calculator_batch_queue_size.share_memory_()
    priority_calculator_p = ctx.Process(
        target=launch_priority_calculator,
        args=(args, priority_calculator_device, training_start_time,
              priority_calculator_batch_queue, priority_calculator_batch_queue_size, priority_update_queue,
              centralizer_mac, target_mac, mixer, target_mixer)
    )
    priority_calculator_p.start()

    buffer_receiver_flag = th.tensor(0)
    buffer_receiver_flag.share_memory_()
    buffer_manager_in_queue = ctx.SimpleQueue()
    buffer_manager_p = ctx.Process(
        target=launch_buffer_manager,
        args=(args, training_start_time, buffer_device, scheme, groups, preprocess, env_info,
              buffer_receiver_flag, buffer_manager_in_queue,
              learner_device, learner_batch_queue, learner_batch_queue_size,
              priority_calculator_device, priority_calculator_batch_queue, priority_calculator_batch_queue_size,
              priority_update_queue)
    )
    buffer_manager_p.start()

    buffer_receiver_out_queue = ctx.SimpleQueue()
    initial_priority_calculator_p = ctx.Process(
        target=launch_initial_priority_calculator,
        args=(args, training_start_time, buffer_device,
              centralizer_mac, target_mac, mixer, target_mixer,
              buffer_receiver_out_queue, buffer_manager_in_queue)
    )
    initial_priority_calculator_p.start()

    buffer_receiver_p = ctx.Process(
        target=launch_buffer_receiver,
        args=(args, training_start_time, buffer_device, scheme, groups, preprocess, env_info,
              centralizer_receiver_queue, buffer_receiver_out_queue, buffer_receiver_flag)
    )
    buffer_receiver_p.start()

    while time.time() - training_start_time < args.training_time:
        continue

    learner_p.join()
    priority_calculator_p.join()
    buffer_manager_p.join()
    initial_priority_calculator_p.join()
    buffer_receiver_p.join()
