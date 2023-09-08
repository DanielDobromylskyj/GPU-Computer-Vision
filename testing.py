import pyopencl as cl
import numpy as np

PLATFORM = cl.get_platforms()[0]
DEVICE = PLATFORM.get_devices()[0]
CTX = cl.Context([DEVICE])
QUEUE = cl.CommandQueue(CTX)

data = np.array([4, 4, 2, 1], dtype=np.float32)

data_buffer = cl.Buffer(CTX, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)



read_data = np.empty_like(data)
cl.enqueue_copy(QUEUE, read_data, data_buffer).wait()

print(read_data)
