# Add my Modules / Packages to path (that pip installs)
import sys
new_path = 'C:\\Users\\danie\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python39\\site-packages'
sys.path.append(new_path)

# Imports (General)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# nn imports
import network
from executor_gpu import Executor as _Executor
import openslide
import random

def LoadTrainingData():
    with open("TrainingData.txt", "r") as f:
        data = eval(f.read())
        
    slide = openslide.OpenSlide("raw/16628.tiff")

    ExtractedData = []
    for index, item in enumerate(data):
        Label, TopLeft = item
        
        region = slide.read_region(TopLeft, 0, (100, 100))
        data = np.asarray(region)[:, :, :3].flatten()

        ExtractedData.append([Label, data])
        
    return ExtractedData


def DrawOutputs(NetPath):
    # Load Network
    TestNetwork = network.Load(NetPath)

    # Init Network Executor
    Executor = _Executor()

    # Load Data
    Data = LoadTrainingData()

    for i in range(len(Data)):
        # Get Random Training Data Sample For Displaying
        RandomLabel, RandomImgData = Data[i]

    
        # Run Network and get what it thinks is the "mico-bac"
        top, left, width, hight = Executor.CalculateOutputs(TestNetwork, RandomImgData)

        print(top, left, width, hight)
        
        # Generate Image From ImgData
        array_shape = (100, 100, 3)
        image_array = RandomImgData.reshape(array_shape)

        # Create a figure and axis using matplotlib
        fig, ax = plt.subplots()
        ax.imshow(image_array)



        # Define the box parameters
        box1_top_left = (top, left)
        box1_width = width
        box1_height = hight
        
        box2_top_left = [RandomLabel[0], RandomLabel[1]]
        box2_width = RandomLabel[2]
        box2_height = RandomLabel[3]

        # Create rectangle patches for the boxes
        rect1 = patches.Rectangle(box1_top_left, box1_width, box1_height, linewidth=2, edgecolor='r', fill=False)
        rect2 = patches.Rectangle(box2_top_left, box2_width, box2_height, linewidth=2, edgecolor='g', fill=False)

        # Add the rectangles to the plot
        ax.add_patch(rect1)
        ax.add_patch(rect2)

        # Set axis limits to the image dimensions
        ax.set_xlim(0, array_shape[1])
        ax.set_ylim(array_shape[0], 0)  # Y-axis is inverted in images

        # Show the plot
        plt.show()


if __name__ == "__main__":
    DrawOutputs("Trained_LargeNetwork_v2.pyn")


