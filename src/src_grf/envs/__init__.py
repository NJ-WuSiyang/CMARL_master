from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from .gfootball import Academy_Counterattack_Hard, Five_vs_Five, Academy_Counterattack_Easy
import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["academy_counterattack_hard"] = partial(env_fn, env=Academy_Counterattack_Hard)
REGISTRY["academy_counterattack_easy"] = partial(env_fn, env=Academy_Counterattack_Easy)
REGISTRY["5_vs_5"] = partial(env_fn, env=Five_vs_Five)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
