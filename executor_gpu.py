import pyopencl as cl
import numpy as np
import math

PLATFORM = cl.get_platforms()[0]
DEVICE = PLATFORM.get_devices()[0]
CTX = cl.Context([DEVICE])
QUEUE = cl.CommandQueue(CTX)


# Try to program CNNs, when I understand them better

class Network:
    size = []
    layers = []
    connections = []
    biases = []
    gradient_data = []

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
        gradient_code = """
                __kernel void CalcGradients(__global float* CurrentLayerGradient, __global float* PreviousLayer, __global float* PreviousLayerGradients, __global float* connections, __global float* BiasSum, float LearningRate) {
                    int i = get_global_id(0);

                    // Get Connection Data
                    int CurrentLayerNode = connections[(i*3)];
                    int PreviousLayerNode = connections[(i*3+1)];
                    float Weight = connections[(i*3+2)];

                    // Get data about Node connection is going to (from in our case)
                    float PrevoisNodeValue = PreviousLayer[PreviousLayerNode];
                    float PreviousNodeGradient = PreviousLayerGradients[PreviousLayerNode];

                    // ReLU Derivative
                    int dYk_dX = 0;
                    if (PrevoisNodeValue > 0) {
                        dYk_dX = 1;
                    }

                    // Update Gradient for next propagation (step)
                    CurrentLayerGradient[CurrentLayerNode] = dYk_dX;

                    // Calculate Gradient of one Connection (bias)
                    float dL_db = (PreviousNodeGradient * dYk_dX);

                    // Multiply dL_db by Weight to get Weight contribution
                    float dl_dW = dL_db * Weight;

                    // Add Gradient to Total (Bias)
                    BiasSum[CurrentLayerNode] += dL_db;

                    // Calculate new weight
                    float NewWeight = Weight - (LearningRate * dl_dW);

                    // Update Weight
                    connections[(i*3 +2)] = NewWeight;
                }
            """

        bias_update_code = """
                __kernel void UpdateBiases(__global float* Biases, __global float* BiasSum, float LearningRate) {
                    int i = get_global_id(0);

                    // Calculate new bias
                    float NewBias = Biases[i] - (LearningRate * BiasSum[i]);

                    // Store bias
                    Biases[i] = NewBias;
                    
                }
            """
        
        self.GradientCalcProgram = cl.Program(CTX, gradient_code).build()
        self.BiasUpdateProgram = cl.Program(CTX, bias_update_code).build()

    def BackPropogateLayer(self, network, layerIndex, learning_rate=0.1):
        """
        Calculates the Gradient Loss of each node in the layer.
        Works on every layer EXCEPT the output layer. <- ?
        """
        # ----- Load ALL needed data into buffers
        
        # Gradients (Returns)
        NewGradientsBuffer = cl.Buffer(CTX, cl.mem_flags.READ_WRITE, size=network.layers[layerIndex].nbytes)
        
        # Previous Layer Node Values
        PrevLayerValuesBuffer = cl.Buffer(CTX, cl.mem_flags.READ_WRITE, size=network.layers[layerIndex + 1].nbytes)
        cl.enqueue_copy(QUEUE, PrevLayerValuesBuffer, network.layers[layerIndex + 1]).wait()

        # Previous Layer Gradients
        PrevLayerGradientsBuffer = cl.Buffer(CTX, cl.mem_flags.READ_WRITE, size=np.array(network.gradient_data).nbytes)
        cl.enqueue_copy(QUEUE, PrevLayerGradientsBuffer, network.gradient_data).wait()

        # Connections
        ConnectionBuffer = cl.Buffer(CTX, cl.mem_flags.READ_WRITE, size=network.connections[layerIndex].nbytes)
        cl.enqueue_copy(QUEUE, ConnectionBuffer, network.connections[layerIndex]).wait()

        # Bias Sum (Returns)
        BiasSumBuffer = cl.Buffer(CTX, cl.mem_flags.READ_WRITE, size=network.layers[layerIndex].nbytes)

        # Bias Buffer
        BiasBuffer = cl.Buffer(CTX, cl.mem_flags.READ_WRITE, size=network.biases[layerIndex].nbytes)
        cl.enqueue_copy(QUEUE, BiasBuffer, network.biases[layerIndex]).wait()

        # ----- Run Programs

        # Run Program For Updating Weights and getting Sum Of Biases Gradients
        self.GradientCalcProgram.CalcGradients(QUEUE, (ConnectionBuffer.size // 3,), None, NewGradientsBuffer, PrevLayerValuesBuffer, PrevLayerGradientsBuffer, ConnectionBuffer, BiasSumBuffer, np.float32(learning_rate)).wait()

        # Run Program To Update Biases
        self.BiasUpdateProgram.UpdateBiases(QUEUE, (ConnectionBuffer.size // 3,), None, BiasBuffer, BiasSumBuffer, np.float32(learning_rate)).wait()

        # ----- Update Network

        # Extract New Values
        NewGradients = np.empty_like(network.layers[layerIndex])
        cl.enqueue_copy(QUEUE, NewGradients, NewGradientsBuffer).wait()

        # Extract New Connections
        NewConnections = np.empty_like(network.connections[layerIndex])
        cl.enqueue_copy(QUEUE, NewConnections, ConnectionBuffer).wait()

        # Extract New Biases
        NewBiases = np.empty_like(network.biases[layerIndex])
        cl.enqueue_copy(QUEUE, NewBiases, BiasBuffer).wait()

        # Set network values
        network.connections[layerIndex] = NewConnections
        network.biases[layerIndex] = NewBiases
        network.gradient_data = NewGradients

    def BackPropogateNetwork(self, network, TargetOutputs, PredicitedOutputs, LearningRate=0.1):
        # Calculate Error and add it to network
        Error = np.array([
            PredicitedOutputs[i] - TargetOutputs[i] for i in range(len(TargetOutputs))
            ])

        network.gradient_data = Error

        # Backpropogate Over the network

        NumberOfLayers = len(network.layers) - 2
        for i in range(len(network.layers)):
            LayerIndex = NumberOfLayers - i

            self.BackPropogateLayer(network, LayerIndex, LearningRate)

        
        


        
