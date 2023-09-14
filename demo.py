import network
import executor_gpu as GpuNet
import numpy as np

# >  Create a new small network
# - (github.com I cant push more than a 100mb file)
# - You can open the .pyn file that this creates and see its format in notepad

# Create a network with 5 inputs, 2 hidden layers of 3 neurons, and 2 outputs
NetworkLayers = [5, 3, 3, 2]

# Generate the network and populate (sparsity * 100)% of connections for memory sake
MyNetwork = network.NetworkMaker(NetworkLayers, sparsity=1).Generate() # Populatre 100% of connections

# Save the network
network.Save(MyNetwork, "DemoNetwork.pyn")


# Load the network
LoadedNetwork = network.Load("DemoNetwork.pyn")

# This class runs the network
Executor = GpuNet.Executor()

# Set some inputs, the Function will raise an error if we dont make it a numpy array
MyInputs = np.array(
    [1, 2, 3, 4, 5]
    )

# Calculate the networks outputs:
MyOutputs = Executor.CalculateOutputs(LoadedNetwork, MyInputs)

# The outputs will likely be all 0.5 due to the connections all being 0 or very close to
print(f"Calculated Outputs: {MyOutputs}")

# Make a label (wanted output for the given input)
Label = np.array([55, 41])

# Run 1 epoche of backpropogation

# Init Class
BackPropagator = GpuNet.BackPropagator()

# Back propogate (1 epoche)
BackPropagator.BackPropogateNetwork(LoadedNetwork, Label, MyOutputs)

# Test
MyOutputs = Executor.CalculateOutputs(LoadedNetwork, MyInputs)
print(f"Calculated Outputs: {MyOutputs} (After 1 Epoche)")




