import network
import executor_gpu as GpuNet
import numpy as np

# Init GPU and Build Programs for Executor
NetworkExecutor = GpuNet.Executor()
NetworkMutator = GpuNet.Mutator()


def TestSingleExecution(net):
    Outputs = NetworkExecutor.CalculateOutputs(net, np.array([3, 4, 1.2, 2, -3]))
    return Outputs

def TestGeneration(size):
    MyNet = network.NetworkMaker(size, sparsity=0.8).Generate()
    return MyNet

def TestMutation(net):
    CopiedNetwork = NetworkMutator.Copy(net)
    NetworkMutator.Mutate(CopiedNetwork)
    return CopiedNetwork

def TestSave(net):
    network.Save(net, "TestNetwork.pyn")

def TestLoad():
    return network.Load("TestNetwork.pyn")

def TestLegacyLoad():
    return network.LegacyLoad("test_legacy_network/Test.NeuralAI")

def TestAll():
    # Make a network with size [5, 4, 2]
    TestNetwork = TestGeneration([5, 4, 2])

    TestSingleExecution(TestNetwork)

    NewNet = TestMutation(TestNetwork)

    TestSingleExecution(NewNet)

    TestSave(NewNet)

    LoadedNet = TestLoad()

    TestSingleExecution(LoadedNet)

    # Test Legacy Loading
    net = TestLegacyLoad()

    NetworkExecutor.CalculateOutputs(net, np.zeros(100))

def StressTest_p1():
    net = TestGeneration([5, 4, 2])
    TimesToTest = 1000
    print(f"Testing Small Network {TimesToTest} Times")
    for i in range(TimesToTest):
        TestSingleExecution(net)
    print(f"Test Finished")

    # Test Larger Network
    net = TestGeneration([5000, 4000, 2000, 50])
    InputData = np.zeros(5000)

    TimesToTest = 300
    print(f"Testing Large Network {TimesToTest} Times")
    for i in range(TimesToTest):
        NetworkExecutor.CalculateOutputs(net, InputData)
    print(f"Test Finished")

def StressTest_p2():
    print("Generating Network")
    net = network.NetworkMaker([5000, 5000, 500, 20], 1).Generate()
    print("Made!")
    network.Save(net, "TESTING.pyn")
    print("Saved")
    network.Load("TESTING.pyn")
    print("Loaded")


if __name__ == "__main__":
    TestAll()
    #StressTest_p1()
    #StressTest_p2()
