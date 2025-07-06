import torch.distributed as dist
import os

def get_rank():
    """获取当前进程的排名，如果未初始化则返回0"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0

def get_world_size():
    """获取分布式环境中的总进程数，如果未初始化则返回1"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1

def is_main_process():
    """检查当前进程是否是主进程 (rank 0)"""
    return get_rank() == 0

def setup_for_distributed(is_master):
    """
    此函数在主进程上禁用打印，除非显式启用。
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
