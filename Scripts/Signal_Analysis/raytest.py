import numpy as np
from numpy.core.records import array
import ray
import scipy.signal
import psutil

num_cpus = psutil.cpu_count(logical=False)
ray.init(num_cpus=num_cpus)


@ray.remote
def f(array_1, array_2):
    # Do some image processing.
    return array_1 + array_2


# Time the code below.

for _ in range(10):
    array_1 = np.zeros((3000, 3000))
    array_2 = np.random((3000, 3000))
    image_id = ray.put(image)
    ray.get([f.remote(image_id, filters[i]) for i in range(num_cpus)])
