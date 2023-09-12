import numpy as np
import random
from concurrent import futures
import os

class Network:
    size = []
    layers = []
    connections = []
    biases = []


def LegacyLoad(path):
    # Get Size and Layers (simple bit)
    with open(path + "_seg1") as metadata_file:
        HeaderData = eval(metadata_file.read())

    NewNet = Network()

    NewNet.size = HeaderData[0]
    NewNet.layers = HeaderData[1]
    # Load Layers
    ConnectionData = []
    for index, size in enumerate(NewNet.size[1:]):
        with open(path + f"_seg2.{index}", "rb") as f:
            data = f.read()

        ConnectionLayer = np.frombuffer(data)
        ConnectionData.append(ConnectionLayer)

    NewNet.connections = ConnectionData

    return NewNet


def ExtractConnections(binaryData, Offset):
    Connections = []

    while True:
        OFFSET_SEGMENT_LENGTH = 10

        SectionLength = int(binaryData[Offset:Offset + OFFSET_SEGMENT_LENGTH].decode().lstrip("0"))  # its complicated
        SectionType = binaryData[Offset + OFFSET_SEGMENT_LENGTH:Offset + OFFSET_SEGMENT_LENGTH + 1].decode()  # What?

        if (SectionType != "c"):
            break

        sub_array_binary = binaryData[
                           Offset + OFFSET_SEGMENT_LENGTH + 1: Offset + OFFSET_SEGMENT_LENGTH + 1 + SectionLength]
        sub_array = np.frombuffer(sub_array_binary, dtype=np.float32)

        Connections.append(sub_array)

        Offset += OFFSET_SEGMENT_LENGTH+1
        Offset += SectionLength

    return Connections, Offset

def ExtractLayers(binaryData, Offset):
    Connections = []

    while (Offset < len(binaryData)):
        OFFSET_SEGMENT_LENGTH = 10

        SectionLength = int(binaryData[Offset:Offset+OFFSET_SEGMENT_LENGTH].decode().lstrip("0")) # its complicated
        SectionType = binaryData[Offset+OFFSET_SEGMENT_LENGTH:Offset+OFFSET_SEGMENT_LENGTH + 1].decode() # What?

        if (SectionType != "l"):
            break

        sub_array_binary = binaryData[Offset + OFFSET_SEGMENT_LENGTH + 1 : Offset + OFFSET_SEGMENT_LENGTH + 1 + SectionLength]
        sub_array = np.frombuffer(sub_array_binary, dtype=np.float32)


        Connections.append(sub_array)

        Offset += OFFSET_SEGMENT_LENGTH+1
        Offset += SectionLength

    return Connections, Offset

def ExtractBiases(binaryData, Offset):
    Connections = []

    while (Offset < len(binaryData)):
        OFFSET_SEGMENT_LENGTH = 10

        SectionLength = int(binaryData[Offset:Offset+OFFSET_SEGMENT_LENGTH].decode().lstrip("0")) # its complicated
        SectionType = binaryData[Offset+OFFSET_SEGMENT_LENGTH:Offset+OFFSET_SEGMENT_LENGTH + 1].decode() # What?

        if (SectionType != "b"):
            break

        sub_array_binary = binaryData[Offset + OFFSET_SEGMENT_LENGTH + 1 : Offset + OFFSET_SEGMENT_LENGTH + 1 + SectionLength]
        sub_array = np.frombuffer(sub_array_binary, dtype=np.float32)


        Connections.append(sub_array)

        Offset += OFFSET_SEGMENT_LENGTH+1
        Offset += SectionLength

    return Connections, Offset

def Load(path):
    NewNet = Network()

    with open(path, "rb") as f:
        BinaryData = f.read()

    LengthInCharsOfSizeData = int(BinaryData[:3].decode().lstrip('0'))

    NetworkSize = eval(BinaryData[3:3 + LengthInCharsOfSizeData].decode())

    Offset = LengthInCharsOfSizeData + 3

    ConnectionData, Offset = ExtractConnections(BinaryData, Offset)
    LayerData, Offset = ExtractLayers(BinaryData, Offset)
    BiasData, Offset = ExtractBiases(BinaryData, Offset)

    NewNet.size = NetworkSize
    NewNet.layers = LayerData
    NewNet.connections = ConnectionData
    NewNet.biases = BiasData

    return NewNet

def ArrayOfNumpyArraysToBytes(array, Identifier:str):
    binaryData = b""

    for sub_array in array:
        binaryArray = sub_array.tobytes()
        metadata = str(len(binaryArray)).zfill(10) + Identifier
        binaryMetadata = metadata.encode()

        binaryData += binaryMetadata + binaryArray

    return binaryData


def Save(network, path):
    NetworkData = str(len(str(network.size))).zfill(3).encode() + str(network.size).encode()

    ConnectionData = ArrayOfNumpyArraysToBytes(network.connections, "c")
    LayerData = ArrayOfNumpyArraysToBytes(network.layers, "l")
    BiasData = ArrayOfNumpyArraysToBytes(network.biases, "b")


    NetworkData += ConnectionData
    NetworkData += LayerData
    NetworkData += BiasData


    Mode = "wb" if os.path.exists(path) else "xb"

    with open(path, Mode) as f:
        f.write(NetworkData)





class NetworkMaker():
    def __init__(self, size, sparsity):
        self.Size = size
        self.sparsity = sparsity



    def Generate(self):
        MyNetwork = Network()
        MyNetwork.size = self.Size
        self.Layers = self.GenerateBlankLayers(self.Size)

        MyNetwork.layers = self.Layers
        MyNetwork.connections = self.GenerateRandomConnections(sparsity=self.sparsity)
        MyNetwork.biases = self.GenerateBiases(self.Size)

        return MyNetwork


    def GenerateBlankLayers(self, size):
        return [
            np.array([0 for i in range(sub_size)], dtype=np.float32)
            for sub_size in size
        ]

    def GenerateBiases(self, size):
        return [
            np.array([0 for i in range(sub_size)], dtype=np.float32)
            for sub_size in size
        ]

    def GenerateRandomConnections(self, sparsity):
        AllConnections = []
        with futures.ThreadPoolExecutor() as executor:
            # Create a list of futures
            futures_list = []
            for layerIndex, layer in enumerate(self.Layers):
                if layerIndex == 0:
                    continue

                future = executor.submit(self.GenerateLayerConnections, layerIndex, layer, sparsity)
                futures_list.append(future)

            # Retrieve the results from the futures
            for future in futures.as_completed(futures_list):
                LayerConnections = future.result()
                AllConnections.append(np.array(LayerConnections, dtype=np.float32))

        return AllConnections

    def GenerateLayerConnections(self, layerIndex, layer, sparsity):
        WEIGHT = 1.0

        LayerConnections = []
        for ActiveNodeIndex, ActiveNode in enumerate(layer):
            for InputNodeIndex, InputNode in enumerate(self.Layers[layerIndex - 1]):
                if random.random() <= sparsity:
                    LayerConnections.extend([ActiveNodeIndex, InputNodeIndex, WEIGHT])

        return LayerConnections

if __name__ == "__main__":
    LayerSizes = [30500 for i in range(20)]
    LayerSizes.append(4)
    LayerSizes.insert(0, 30000)


    network = NetworkMaker(LayerSizes, 0.005).Generate()
    print("Starting Save")
    
    Save(network, "SmallNetwork.pyn")




