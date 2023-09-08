import pyopencl as cl
import numpy as np
import math

PLATFORM = cl.get_platforms()[0]
DEVICE = PLATFORM.get_devices()[0]
CTX = cl.Context([DEVICE])
QUEUE = cl.CommandQueue(CTX)


class Network:
    size = []
    layers = []
    connections = []

# Errors

class BadInput(Exception): ...
class BadNetwork(Exception): ...

# Executor


class Executor:
    def __init__(self):
        layer_program = """
                __kernel void calc_layers(__global float* inputLayer, __global float* connectionBuffer, __global float* outputLayer) {
                    int GlobalIndex = get_global_id(0);

                    // Get the correct indexes and everything for the calculations
                    int OutputIndex = connectionBuffer[(GlobalIndex * 4)];
                    int InputIndex = connectionBuffer[(GlobalIndex * 4) + 1];
                    int Weight = connectionBuffer[(GlobalIndex * 4) + 2];
                    int Bias = connectionBuffer[(GlobalIndex * 4) + 3];

                    // Calculate new value
                    int Value = inputLayer[InputIndex];
                    Value = (Value * Weight) + Bias;

                    // Write New Value To LayerBuffer
                    outputLayer[OutputIndex] = Value;
                }
            """

        activation_program = """
                __kernel void SigmoidLayer(__global float* Layer) {
                    int i = get_global_id(0);
                    
                    float exp_val = exp(-Layer[i]);
                    Layer[i] = 1.0f / (1.0f + exp_val);
                }
            """

        self.ConnectionProgram = cl.Program(CTX, layer_program).build()
        self.ActivationProgram = cl.Program(CTX, activation_program).build()

    def CalculateOutputs(self, network, inputs):
        # Run some quick checks
        if type(inputs) != np.ndarray:
            raise BadInput(f"Inputs must be a numpy array, not {type(inputs)}")

        if network.size[0] != len(inputs):
            NetIsLarger = network.size[0] > len(inputs)
            raise BadInput(f"{'Not enough' if NetIsLarger else 'Too many'} 'Inputs'. Have {len(inputs)}, Need {network.size[0]}")

        if network == None:
            raise BadNetwork(f"'Network' Is a NoneType variable. Please parse a network")

        if len(network.layers) == 0:
            raise BadNetwork("'Network' Is Empty (No Layer Data)")

        # Load Inputs to buffer
        InputBuffer = cl.Buffer(CTX, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=inputs)

        for layerIndex, layer in enumerate(network.layers):
            if layerIndex == 0: # No connections for first layer, saves some time + memory
                continue

            OutputBuffer = cl.Buffer(CTX, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=network.layers[layerIndex])

            # Load connections to a buffer
            Connections = network.connections[layerIndex - 1]
            ConnectionBuffer = cl.Buffer(CTX, cl.mem_flags.READ_WRITE, size=Connections.nbytes)
            cl.enqueue_copy(QUEUE, ConnectionBuffer, Connections).wait()

            # Calculate the outputs and save them to 'OutputBuffer'
            OutputBuffer = self.RunLayer(InputBuffer, OutputBuffer, ConnectionBuffer)

            # Set Outputs to Inputs for next layer

            InputBuffer = OutputBuffer



        outputs = np.empty_like(layer)
        cl.enqueue_copy(
            QUEUE,
            outputs,
            OutputBuffer
        ).wait()

        return outputs



    def RunLayer(self, InputBuffer, OutputBuffer, ConnectionBuffer):
        global_size = ConnectionBuffer.size // 4
        self.ConnectionProgram.calc_layers(QUEUE, (global_size,), None, InputBuffer, ConnectionBuffer, OutputBuffer).wait()

        # Run the activation function on outputs
        self.ActivationProgram.SigmoidLayer(QUEUE, (OutputBuffer.size,), None, OutputBuffer).wait()

        return OutputBuffer


# Mutator
from copy import deepcopy
import random

class Mutator:
    def __init__(self):
        mutate_code = """
                __kernel void Mutate(__global float* Connections, int seed, int mutate_chance) {
                    int index = get_global_id(0);

                    // Generate Random Number between 0 and 'size'
                    const int size = 50;

                    // Initialize state based on seed and index
                    uint state = seed + index;

                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;

                    int x = state % 50;
                    
                    if (x < mutate_chance) {
                    
                        // Generate random bias and weight values
                        float Bias = (state % size);
                        float Weight = (state % size);

                        Bias = (Bias - 25) / 10;
                        Weight = (Weight - 25) / 10;


                        int connIndex = index * 4;
                        Connections[connIndex] = Connections[connIndex];
                        Connections[connIndex + 1] = Connections[connIndex + 1];
                        Connections[connIndex + 2] = Connections[connIndex + 2] + Weight;
                        Connections[connIndex + 3] = Connections[connIndex + 3] + Bias;
                    }
                }
        """
        self.MutateProgram = cl.Program(CTX, mutate_code).build()

        # Set Some Settings Of MutateProgram
        self.MutateProgram.Mutate.set_scalar_arg_dtypes([None, np.int32, np.int32])

        # Some Defaults
        self.MaxConnectionToBeModified = 20

    def Copy(self, network):
        return deepcopy(network)

    def Mutate(self, network, mutate_chance=3):
        self.CreateNewConnections(network)
        self.DeleteOldConnections(network) # This needs to be sped up
        self.MutateExistingLayers(network, mutate_chance)


    def CreateNewConnections(self, network):
        for index, ConnectionLayer in enumerate(network.connections):
            NumberOfConnectionsToCreate = min([math.ceil(ConnectionLayer.size * 0.01), self.MaxConnectionToBeModified])
            for i in range(NumberOfConnectionsToCreate):
                InsertIndex = random.randint(0, (len(network.connections[index]) -1) // 4) * 4

                # Generate blank connection
                ConnectionLayer = np.insert(ConnectionLayer, InsertIndex, 0)
                ConnectionLayer = np.insert(ConnectionLayer, InsertIndex, 1)
                ConnectionLayer = np.insert(ConnectionLayer, InsertIndex, 0)
                ConnectionLayer = np.insert(ConnectionLayer, InsertIndex, 0)

    def DeleteOldConnections(self, network):
        for ConnectionLayer in network.connections:
            NumberOfConnectionsToDelete = min(math.ceil(ConnectionLayer.size * 0.01), self.MaxConnectionToBeModified)
            if NumberOfConnectionsToDelete > 0:
                delete_indices = random.sample(range(len(ConnectionLayer)), NumberOfConnectionsToDelete)
                ConnectionLayer = np.delete(ConnectionLayer, delete_indices)


    def MutateExistingLayers(self, network, mutate_chance):
        for index, ConnectionLayer in enumerate(network.connections):
            ConnBuffer = cl.Buffer(CTX, cl.mem_flags.READ_WRITE, size=ConnectionLayer.nbytes)
            cl.enqueue_copy(QUEUE, ConnBuffer, ConnectionLayer).wait()

            # A Set seed for the random function within the gpu script
            seed = random.randint(0, 500)

            # Mutate Connections
            self.MutateProgram.Mutate(QUEUE, (ConnBuffer.size // 4,), None, ConnBuffer, np.int32(seed), np.int32(mutate_chance)).wait()

            # Read the new Connections
            output_buffer = np.empty((ConnectionLayer.size,), dtype=np.float32)
            cl._enqueue_read_buffer(QUEUE, ConnBuffer, output_buffer)
            network.connections[index] = output_buffer



