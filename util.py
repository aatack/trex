import time


def benchmark(f, repeats=1):
    """Time the execution of a function, averaged over multiple runs."""
    times = []
    for _ in range(repeats):
        start_time = time.time()
        f()
        end_time = time.time()
        times.append(end_time - start_time)
    return sum(times) / len(times)
