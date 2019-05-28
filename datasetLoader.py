import csv
import numpy as np
import random
import math

"""

This function opens hour.csv file from Concrete_Data and returns
tuple containing training and testing data in such manner as ndarray:
resultTrainingData[0] - arrays of inputs
resultTrainingData[1] - outputs
"""
"""
Returns tuple containing training and testing data for sin function in such manner as ndarray:
resultTrainingData[0] - arrays of inputs
resultTrainingData[1] - outputs
"""

def readConcreteDataset():
    # open csv file
    with open('Concrete_Data.csv', newline='') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',', quotechar='"')
        # read number of rows
        row_count = sum(1 for row in csvReader)
        # come back at beginning of file
        csvFile.seek(0)
        # generate random indexes of testing data - it will be 10% of total data
        testDataIndexes = random.sample(range(1, row_count), int(row_count / 10))
        # initialize data lists
        testData = []
        trainingData = []

        # ommit first row with info data
        firstLine = next(csvReader)

        idx = 1
        maximums = [-1 * 8]
        # distribute data between test and training lists
        for row in csvReader:
            if idx in testDataIndexes:
                testData.append(row)
            else:
                trainingData.append(row)
            idx = idx + 1
        print(maximums)
        idx+= 1
        # return tuples of training and test data

        resultTrainingData = (
        np.array(normalizeConcreteInputDataTable(trainingData)), np.array([float(row[8])  for row in trainingData]))
        resultTestData = (
        np.array(normalizeConcreteInputDataTable(testData)), np.array([float(row[8])  for row in testData]))
        return (resultTrainingData, resultTestData)



def normalizeConcreteInputDataTable(dataTable):
    return [[(float(row[0]))/540 -0.5, (float(row[1]))/359.4 -0.5, (float(row[2]))/200.1 -0.5,
             float(row[3])/247 -0.5, (float(row[4]))/32.2 -0.5, float(row[5])/1145 -0.5, (float(row[6]))/992.6 -0.5, (float(row[7]))/365 -0.5] for row in dataTable]

def getTestDataSet(size):
    trainInputs = []
    trainOutputs = []
    testInputs = []
    testOutputs = []
  
    for i in range(int(size*0.8)):
        s = np.random.random_integers(1, 10, 2)
        trainInputs.append(s)
        if s[1] > 5.0:
            trainOutputs.append(1)
        else:
            trainOutputs.append(0)

    for i in range(int(size*0.2)):
        s = np.random.random_integers(1,10,2)
        testInputs.append(s)
        if s[1] > 5.0:
            testOutputs.append(1)
        else:
            testOutputs.append(0)

    return ((np.array(trainInputs), np.array(trainOutputs)),
            (np.array(testInputs), np.array(testOutputs)))
