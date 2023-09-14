import network
import executor_gpu as GpuNet
import numpy as np
import time
import random

# Special Case Import(s)
import openslide

# Helpfull Resources: https://neptune.ai/blog/backpropagation-algorithm-in-neural-networks-guide


Executor = GpuNet.Executor()
BackPropagator = GpuNet.BackPropagator()

def EpocheReport(EpocheNumber, MaxEpoches, TimeTaken):
    print("\n")
    print(f"Completed Epoche {EpocheNumber} Out of {MaxEpoches}. ({round(EpocheNumber/MaxEpoches*100)}%)")
    print(f"Took {TimeTaken}s. Time Left: {round((TimeTaken*(MaxEpoches-EpocheNumber))/60, 2)} mins")

def Teach(network, TrainingData, Epoches, LearningRate=0.1, ReportAfterEpoche=True, UpdateAfterEachItem=False):
    for epoche in range(Epoches):
        start_time = time.time()

        for item in TrainingData:
            Label, NetworkInputs = item

            # Test network on data
            NetworkOutputs = Executor.CalculateOutputs(network, NetworkInputs)

            # Update Weights and biases
            BackPropagator.BackPropogateNetwork(network, Label, NetworkOutputs, LearningRate=LearningRate)

            if UpdateAfterEachItem:
                print("Completed One Back Propagation")

        elapsed_time = time.time() - start_time
        if ReportAfterEpoche:
            EpocheReport( epoche, Epoches, elapsed_time)
            

    return network
            

        


if __name__ == "__main__":
    def LoadTrainingData(numberOfItems):
        # Open File and Read Data
        with open("TrainingData.txt", "r") as f:
            data = eval(f.read())

        # Randomly Select some data
        numberOfItems = min([numberOfItems, len(data) - 1])
        RandomData = random.choices(data, k=numberOfItems)

        # Open the a slide with microbac on it
        slide = openslide.OpenSlide("raw/16628.tiff")

        # Loop over the data and load it once so we dont have to keep doing it
        ExtractedData = []
        for item in RandomData:
            Label, TopLeft = item

            region = slide.read_region(TopLeft, 0, (100, 100))
            data = np.asarray(region)[:, :, :3].flatten()

            ExtractedData.append([Label, data])

        return ExtractedData

    # Load the blank Network
    MyNetwork = network.Load("LargeNetwork.pyn")
    
    # Load some training Data
    TrainingData = LoadTrainingData(numberOfItems=200)


    #                            Epoches
    Teach(MyNetwork, TrainingData, 100)
    

    # Save the network
    network.Save(MyNetwork, "Trained_LargeNetwork.pyn")

    print("COMPLETE") # DEBUG

