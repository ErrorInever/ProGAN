import time


def print_epoch_time(f):
    """Calculate time of each epoch and print it"""
    def timed(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print("epoch time: %2.1f min" % ((te-ts)/60))
        return result
    return timed
