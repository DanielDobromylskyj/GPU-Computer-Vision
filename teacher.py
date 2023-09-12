import network
import executor_gpu as GpuNet
import random
import numpy as np
from copy import deepcopy
import time

# Special Case Import(s)
import openslide

# Helpfull Resources: https://neptune.ai/blog/backpropagation-algorithm-in-neural-networks-guide


Executor = GpuNet.Executor()


class Trainer:
    def __init__(self, Network, ErrorFunc, TraingingData):
        self.Network = Network
        
        self.TrainingData = TraingingData
        self.ErrorFunc = ErrorFunc

        self.TestsPerScore = 20
        self.NumberOfCycles = 100

        self.StepSize = 0.5

    def ScoreNetwork(self, network):
        """ Score a network multiple times and take an average """
        Total = 0
        for i in range(self.TestsPerScore):
            Total += self.TestNetwork(network, random.choice(self.TrainingData))

        return Total / self.TestsPerScore
        
    def TestNetwork(self, network, sample):
        """ Score a single network once. """
        label, data = sample

        outputs = Executor.CalculateOutputs(network, data)

        error = sum([abs(x) for x in outputs])
        return error

    def Iterate(self):
        BestScore = self.ScoreNetwork(self.Network)

        Connections = np.array(self.Network.connections)
        
        for layerIndex, layer in enumerate(self.Network.layers):
            if layerIndex == 0:
                continue


            for connectionIndex in range(len(self.Network.connections[layerIndex-1]) // 3):
                ConnIndex = (connectionIndex * 3) + 2
                Connections = np.array(self.Network.connections[layerIndex-1])
                
                # Try increasing the value of the weight
                Connections[ConnIndex] += self.StepSize

                Score = self.ScoreNetwork(self.Network)
                if abs(Score) > abs(BestScore):
                    # Try going 1 step size in the other direction
                    Connections[ConnIndex] -= self.StepSize * 2

                    Score = self.ScoreNetwork(self.Network)
                    if abs(Score) > abs(BestScore):
                        # If nothing worked, reset to old values
                        Connections[ConnIndex] += self.StepSize

                    else:
                        BestScore = Score
                else:
                    BestScore = Score

        self.Network.connections = Connections
            

                
    
    def Run(self):
        Cycle = 0

        TotalTime = 0
        while Cycle < self.NumberOfCycles:
            start = time.time()
            
            self.Iterate()

            elapsed = time.time() - start
            TotalTime += elapsed
            print(f"> Cycles Number:{Cycle} Out of {self.NumberOfCycles}")
            print(f"- Time Taken: {elapsed}")
            print(f"- Estimate Time Left: {(TotalTime / Cycle) * self.NumberOfCycles}")
            
            Cycle += 1

        return self.Network


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


    def ErrorFunc(outputs, labelData):
        # Unpack Data
        WantedX, WantedY, WantedWidth, WantedHight = labelData

        OutputX, OutputY, OutputWidth, OutputHight = outputs

        # Calculate The Variation Between Output and Real

        return [(WantedX / 100) - OutputX,
                (WantedY / 100) - OutputY,
                (WantedWidth / 100) - OutputWidth,
                (WantedHight / 100) - OutputHight]

    # Load the blank Network
    net = network.Load("SmallNetwork.pyn")
    
    # Load some training Data
    TrainingData = LoadTrainingData(numberOfItems=200)

    # Initialize the Trainer with its needed inputs
    trainer = Trainer(net, ErrorFunc, TrainingData)

    print("Starting Training")

    # Run the trainer and get the trained network back
    TrainedNet = trainer.Run()

    # Save the network
    network.Save(TrainedNet, "Trained_SmallNetwork.pyn")

    print("COMPLETE") # DEBUG

