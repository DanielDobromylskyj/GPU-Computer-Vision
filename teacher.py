import network
import executor_gpu as GpuNet
import random
import numpy as np

# Special Case Import(s)
import openslide

Executor = GpuNet.Executor()
Mutator = GpuNet.Mutator()

class Trainer:
    def __init__(self, Network, FitnessFunc, TraingingData):
        self.BestNetwork = Network
        self.BestFitness = -1 # we set this to -1 as we don't know the fitness, so we just set it to a low number

        self.TrainingData = TraingingData
        self.FitFunc = FitnessFunc

        self.TARGET_FITNESS = "inf"
        self.MAX_TEACHING_CYCLES = 500
        self.BATCH_SIZE = 5
        self.TESTS_PER_NETWORK = 5

        # Smart Score
        self.ITERATE_SCORE = 100
        self.NumberOfItemsToScore = 1


    def Mutate(self, network):
        NewNetwork = Mutator.Copy(network)
        Mutator.Mutate(NewNetwork)
        return NewNetwork
    
    def SmartScore(self, network):
        TrainingData = self.TrainingData[0:self.NumberOfItemsToScore]

        cumulativeScore = 0
        for chunk in TrainingData:
            cumulativeScore += self._SmartScore(network, chunk)

        averageScore = cumulativeScore / self.NumberOfItemsToScore

        if averageScore > self.ITERATE_SCORE:
            print(f"Iterate Score Reached, traing on {self.NumberOfItemsToScore + 1} Data")
            # If we run out of data to train on, return "fin"
            if (self.NumberOfItemsToScore + 1) > len(self.TrainingData):
                return net
            
            self.NumberOfItemsToScore += 1

        return averageScore


    def _SmartScore(self, network, trainingData):
        label, data = trainingData
        
        Outputs = Executor.CalculateOutputs(network, data)
        result = self.FitFunc(Outputs, label)
        
        return result
    

    def Score(self, network):
        TotalScore = 0
        for TestNumber in range(self.TESTS_PER_NETWORK):
            TotalScore += self.TestNetwork(network)

        return TotalScore / self.TESTS_PER_NETWORK # Get average

    def TestNetwork(self, network):
        label, data = random.choice(self.TrainingData)

        # Calculate Outputs
        Outputs = Executor.CalculateOutputs(network, data)

        # Run fitness function on results
        result = self.FitFunc(Outputs, label)
        return result

    def Run(self, SaveEvery=3):
        self.BestFitness = self.SmartScore(self.BestNetwork)

        teachingCycle = 0
        NextSave = SaveEvery
        while (teachingCycle < self.MAX_TEACHING_CYCLES):
            for BatchNumber in range(self.BATCH_SIZE):
                net = self.Mutate(self.BestNetwork)
                #score = self.Score(net)
                score = self.SmartScore(net)


                if score > self.BestFitness:
                    self.BestNetwork = net
                    self.BestFitness = score


            if self.TARGET_FITNESS != "inf":
                if (self.BestFitness > self.TARGET_FITNESS):
                    return self.BestNetwork

            print("Fitness:", self.BestFitness)

            if teachingCycle == NextSave:
                network.Save(self.BestNetwork, "AutoSave.pyn")
                NextSave += SaveEvery

            teachingCycle += 1
        return self.BestNetwork


if __name__ == "__main__":
    def LoadTrainingData(numberOfItems):
        print("Loading Data...")
        with open("TrainingData.txt", "r") as f:
            data = eval(f.read())

        print("Selecting Data...")
        numberOfItems = min([numberOfItems, len(data) - 1])
        RandomData = random.choices(data, k=numberOfItems)

        print("Loading Image...")
        slide = openslide.OpenSlide("raw/16628.tiff")

        print("Extracting Data...")
        ExtractedData = []
        for index, item in enumerate(RandomData):
            Label, TopLeft = item

            region = slide.read_region(TopLeft, 0, (100, 100))
            data = np.asarray(region)[:, :, :3].flatten()

            ExtractedData.append([Label, data])

            print(f"Extracted Item {index} of {numberOfItems} -> {round((index / numberOfItems) * 100)}%")

        return ExtractedData


    def TestFitFunc(outputs, labelData):
        # Unpack Data
        WantedX, WantedY, WantedWidth, WantedHight = labelData

        OutputX, OutputY = outputs[0] * 100, outputs[1] * 100
        OutputWidth, OutputHight = outputs[2] * 100, outputs[3] * 100

        # Calculate The Variation Between Output and Real
        Variation = 0
        Variation += abs(OutputX - WantedX) * 10  # Make this more important
        Variation += abs(OutputY - WantedY) * 10  # Make this more important
        Variation += abs(OutputWidth - WantedWidth)  # Less important
        Variation += abs(OutputHight - WantedHight)  # Less important

        return 2200 - Variation  # 2200 is the best possible score


    Data = LoadTrainingData(numberOfItems=200)


    net = network.Load("LargerNetwork.pyn")# BaseNetwork.pyn

    trainer = Trainer(net, TestFitFunc, Data)

    # Set some safety / Target constants (Smart Scoring Enabled)
    trainer.TARGET_FITNESS = "inf" # Needs to be infinity for smart scoring
    trainer.ITERATE_SCORE = 2100 # Also needed for smart scoring
    
    trainer.MAX_TEACHING_CYCLES = 5000
    trainer.BATCH_SIZE = 10
    

    print("Training")

    TrainedNet = trainer.Run()

    network.Save(TrainedNet, "Trained_v3.pyn")

    print("COMPLETE")

