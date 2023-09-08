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
    [1, 2, 3, 4, -5]
    )

# Calculate the networks outputs:
MyOutputs = Executor.CalculateOutputs(LoadedNetwork, MyInputs)

# The outputs will likely be all 0.5 due to the connections all being 0 or very close to
print(f"Calculated Outputs: {MyOutputs}")

# Mutater class
Mutator = GpuNet.Mutator()

# Copy network
NewNetwork = Mutator.Copy(LoadedNetwork)

# Mutate network, as this is a small network, and my module is for large networks,
# we must change the chance for a connection to mutate. (default=6%)

MutateChance = 50

# Mutate
Mutator.Mutate(NewNetwork, MutateChance)

# Recalculate the results
NewOutputs = Executor.CalculateOutputs(NewNetwork, MyInputs)

print(f"Mutated Network Outputs: {NewOutputs}")







