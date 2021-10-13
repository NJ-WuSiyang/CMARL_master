import os
import time as time
import torch as th
import torch.multiprocessing as mp

from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from runners.replay_runner import EpisodeRunner
from modules.container_qs import ContainerQs
from centralizer.main import launch as launch_centralizer
from container.main import launch as launch_container


def run(args, logger):
    args.batch_size_run = 1

    model_root_path = os.path.join(args.save_model_path, "models")
    for model_path, dirs, files in os.walk(model_root_path):
        print(model_path)
        for file in files:
            if file != 'agent.th':
                os.remove(os.path.join(model_path, file))
        if len(files) != 1:
            continue
        # Init tester so we can get env info
        if args.env == "sc2":
            args.env_args["replay_dir"] = model_path
        else:
            args.env_args["logdir"] = model_path
            # args.env_args["render"] = True
            args.env_args["dump_freq"] = 1
            args.env_args["write_full_episode_dumps"] = True
            args.env_args["write_goal_dumps"] = True

        tester = EpisodeRunner(args=args, logger=logger)
        tester.args.device = "cuda"

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

        mac = mac_REGISTRY[args.mac](buffer_scheme, groups, args)
        mac.load_models(model_path)
        mac.cuda()
        tester.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

        for _ in range(1):
            tester.run(test_mode=True)
        if args.env == "sc2":
            tester.save_replay()

        tester.close_env()
    return
