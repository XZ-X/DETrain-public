
import functools

def register_iter(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        value = func(*args, **kwargs)
        return value
    if func is None:
        return func
    return wrapper_decorator

def next_batch(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        value = func(*args, **kwargs)
        return value
    if func is None:
        return func
    return wrapper_decorator


