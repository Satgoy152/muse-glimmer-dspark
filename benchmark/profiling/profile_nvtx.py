"""Low-overhead CPU launch annotations; deliberately no CUDA synchronization."""
from functools import wraps


def range_call(label):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            import torch
            name = label
            if label == "profile.graph":
                name = ("profile.draft_graph" if "DFlash" in type(args[0]).__name__
                        else "profile.target_graph")
            with torch.cuda.nvtx.range(name):
                return function(*args, **kwargs)
        return wrapped
    return decorate
