import os
import time as time
import torch as th
import torch.multiprocessing as mp

from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from runners.parallel_runner import ParallelRunner
from modules.container_qs import ContainerQs
from centralizer.main import launch as launch_centralizer
from container.main import launch as launch_container


def run_sequential(args, logger):
    mp.set_start_method("spawn")
    ctx = mp.get_context("spawn")

    training_start_time = time.time()
    t_env = th.tensor(0)
    t_env.share_memory_()

    # Init tester so we can get env info
    tester = ParallelRunner(args=args, logger=logger)

    # Set up schemes and groups here
    env_info = tester.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    # get the scheme of the buffer
    _buffer = ReplayBuffer(scheme, groups, 1, env_info["episode_limit"] + 1,
                           preprocess=preprocess,
                           device="cpu")
    buffer_scheme = _buffer.scheme

    tester.setup(scheme=scheme, groups=groups, preprocess=preprocess,
                 mac=mac_REGISTRY[args.mac](buffer_scheme, groups, args), device="cpu")

    centralizer_mac = mac_REGISTRY[args.mac](buffer_scheme, groups, args)
    centralizer_mac.share_memory()
    centralizer_mac_time_step = th.tensor(0)
    centralizer_mac_time_step.share_memory_()
    centralizer_receiver_queue = ctx.SimpleQueue()

    container_qs = ContainerQs(args)
    container_qs.share_memory()
    container_macs = []
    for i in range(args.n_container):
        container_mac = mac_REGISTRY[args.mac](buffer_scheme, groups, args)
        container_qs.load(i, container_mac.agent.fc2)
        container_macs.append(container_mac)

    centralizer_p = ctx.Process(
        target=launch_centralizer,
        args=(ctx, args, scheme, groups, preprocess, env_info,
              args.centralizer_learner_device, args.centralizer_priority_calculator_device, args.centralizer_buffer_device,
              centralizer_receiver_queue,
              training_start_time, t_env,
              centralizer_mac, centralizer_mac_time_step)
    )
    centralizer_p.start()

    container_ps = []
    for i in range(args.n_container):
        container_p = ctx.Process(
            target=launch_container,
            args=(ctx, args, i, scheme, groups, preprocess, env_info,
                  args.container_learner_device[i], args.container_priority_calculator_device[i], args.container_buffer_device[i], args.container_actor_devices[i],
                  centralizer_receiver_queue,
                  training_start_time, t_env,
                  centralizer_mac, container_macs[i], container_qs,
                  centralizer_mac_time_step)
        )
        container_p.start()

    last_log_time = training_start_time
    last_test_time = training_start_time
    last_save_model_time = training_start_time
    logger.console_logger.info("Beginning training for {} minutes".format(args.training_time / 60.0))

    while time.time() - training_start_time < args.training_time:
        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // tester.batch_size)
        if time.time() - last_test_time >= args.test_time_interval:
            logger.console_logger.info("t_env: {}".format(int(t_env)))
            logger.console_logger.info("time passed: {} minutes".format((time.time() - training_start_time) / 60.0))
            last_test_time = time.time()

            tester.t_env = int(t_env)
            tester.mac.load_state(centralizer_mac)
            for _ in range(n_test_runs):
                tester.run(test_mode=True)

        if time.time() - last_log_time >= args.log_time_interval:
            last_log_time = time.time()

            logger.log_stat("time_passed", last_log_time - training_start_time, int(t_env))
            logger.print_recent_stats()

        if args.save_model and time.time() - last_save_model_time > args.save_model_time_interval:
            last_save_model_time = time.time()
            save_path = os.path.join(args.save_model_path, "models", str(round(last_save_model_time - training_start_time)))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))
            centralizer_mac.save_models(save_path)

    centralizer_p.join()
    for container_p in container_ps:
        container_p.join()

    tester.close_env()
