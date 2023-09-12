import pyopencl as cl
import numpy as np
import math

PLATFORM = cl.get_platforms()[0]
DEVICE = PLATFORM.get_devices()[0]
CTX = cl.Context([DEVICE])
QUEUE = cl.CommandQueue(CTX)


# NOTE: Switch To ReLU Function,
# Try to program CNNs, when I understand them better

class Network:
    size = []
    layers = []
    connections = []
    biases = []

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
                    int OutputIndex = connectionBuffer[(GlobalIndex * 3)];
                    int InputIndex = connectionBuffer[(GlobalIndex * 3) + 1];
                    int Weight = connectionBuffer[(GlobalIndex * 3) + 2];

                    // Calculate new value
                    int Value = inputLayer[InputIndex];
                    Value = (Value * Weight);

                    // Write New Value To LayerBuffer
                    outputLayer[OutputIndex] = Value;
                }
            """

        activation_program = """
                __kernel void SigmoidLayerAndBias(__global float* Layer, __global float* Bias) {
                    // ACTIVATION FUNCTION
                    int i = get_global_id(0);

                    // Get value and add its Bias
                    float value = Layer[i] + Bias[i];

                    // SIGMOID
                    //float exp_val = exp(-value);
                    //Layer[i] = 1.0f / (1.0f + exp_val);

                    // ReLU
                    Layer[i] = max(0.0f, value);
                }
            """



        self.ConnectionProgram = cl.Program(CTX, layer_program).build()
        self.ActivationProgram = cl.Program(CTX, activation_program).build()

    def CalculateOutputs(self, network, inputs, ForBackpropagation=False):
        # Run some quick checks
        if type(inputs) != np.ndarray:
            raise BadInput(f"Inputs must be a numpy array, not {type(inputs)}")

        if network.size[0] != len(inputs):
            NetIsLarger = network.size[0] > len(inputs)
            raise BadInput(f"{'Not enough' if NetIsLarger else 'Too many'} 'Inputs'. Have {len(inputs)} inputs, Need {network.size[0]}.")

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

            # Load Biases into a buffer
            Biases = network.connections[layerIndex - 1]
            BiasesBuffer = cl.Buffer(CTX, cl.mem_flags.READ_WRITE, size=Biases.nbytes)
            cl.enqueue_copy(QUEUE, BiasesBuffer, Biases).wait()

            # Calculate the outputs and save them to 'OutputBuffer'
            OutputBuffer = self.RunLayer(InputBuffer, OutputBuffer, ConnectionBuffer, BiasesBuffer)

            # Store outputs if needed
            if ForBackpropagation:
                # Make empty array
                LayerValues = np.empty_like(network.layers[layerIndex])

                # Fill array
                cl.enqueue_copy(
                    QUEUE,
                    LayerValues,
                    OutputBuffer
                ).wait()

                # Store array
                network.layers[layerIndex] = LayerValues

                

            # Set Outputs to Inputs for next layer
            InputBuffer = OutputBuffer



        outputs = np.empty_like(layer)
        cl.enqueue_copy(
            QUEUE,
            outputs,
            OutputBuffer
        ).wait()

        return outputs



    def RunLayer(self, InputBuffer, OutputBuffer, ConnectionBuffer, BiasBuffer):
        global_size = ConnectionBuffer.size // 4
        self.ConnectionProgram.calc_layers(QUEUE, (global_size,), None, InputBuffer, ConnectionBuffer, OutputBuffer).wait()

        # Run the activation function on outputs
        self.ActivationProgram.SigmoidLayerAndBias(QUEUE, (OutputBuffer.size,), None, OutputBuffer, BiasBuffer).wait()

        return OutputBuffer

# Training / Backprop

class BackPropagator:
    def __init__(self):
        program_code = """
                __kernel void SigmoidLayerAndBias(__global float* Layer, __global float* Previous) {
                    // ACTIVATION FUNCTION
                    int i = get_global_id(0);

                    // Get value and add its Bias
                    float value = Layer[i] + Bias[i];

                    // SIGMOID
                    //float exp_val = exp(-value);
                    //Layer[i] = 1.0f / (1.0f + exp_val);

                    // ReLU
                    Layer[i] = max(0.0f, value);
                }
            """



        self.Program = cl.Program(CTX, program_code).build()
