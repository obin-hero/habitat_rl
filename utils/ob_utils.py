import time
def log_time(prev_time=None, log='', return_time=False):
    if prev_time is not None :
        delta = time.time() - prev_time
        print("[TIME] ", log, delta)
    if return_time:
        return time.time(), delta
    else:
        return time.time()