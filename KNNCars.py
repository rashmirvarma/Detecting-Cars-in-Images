'''
Created on Dec 4, 2017

@author: Chinmay Mishra
'''

from PIL import Image
from collections import Counter
import os
from random import shuffle

class Agent():
    '''
    Agent class which accepts images from the user and classifies them.
    '''
    def __init__(self):
        self._TrainingFeatureList = None
        self._histData = None
        self._featureList = []
        self._kValue = 3
        pass
    
    def setFeatureLists(self,featureList):
        '''
        This sets the training features for the agent
        '''
        self._TrainingFeatureList = featureList
        
    def TestUserInput(self):
        '''
        Function to take input image file path and k value
        '''
        imageFile = raw_input("Enter test image file path: ")
        if not imageFile.split('.')[-1] in ["jpeg","png","tiff","png","bmp","jpg"]:
            raise Exception("Entered file path is not an image.")
        else:
            im = Image.open(imageFile)
            self._histData = im.histogram()
            featureList = self.extractFeatures()
            distanceList = self.findDistance(featureList)
            return self.findClass(distanceList, 3)
    
    def findClass(self,distanceList,kvalue = None):
        '''
        function to find the most common class among the k closest neighbors
        '''
        if kvalue is None:
            kvalue = self._kValue
        clsCount = Counter()
        for i in distanceList[0:kvalue+1]:
            clsCount[i[0]]+=1
        return clsCount.most_common(1)[0][0]
    
    
    def extractFeatures(self):
        '''
        Function to extract features from the list
        '''
        featureList = [0]*48
        for ind,rgbintensityData in enumerate(self._histData):
            rgbIndex = ind/256
            rgbshade = (ind%256)/16
            featureList[rgbIndex*16+rgbshade] += rgbintensityData
        return featureList
    

    def findDistance(self,featureList=None):
        '''
        Function to find data from the list
        '''
        distanceList = []
        for feature in self._TrainingFeatureList:
            sumSquaredDistance = 0
            for j in range(48):
                try:
                    sumSquaredDistance += (float(featureList[j]) - float(feature[j]))**2
                except:
                    print j,featureList[j], feature[j]
            distanceList.append((feature[-1],sumSquaredDistance))
        distanceList = sorted(distanceList,key=lambda x: x[1])
        return distanceList
    
class Environment():
    '''
    Environment class passes the folder containing 
    '''
    def __init__(self):
        self._agent = Agent()
        pass
    
    
    def loadFeaturesFromList(self,imageList):
        featureList = []
        for fileName in imageList:
            imgFeatureList = [0]*48
            if fileName.split('.')[-1] in ["jpeg","png","tiff","png","bmp","jpg"]:
                im = Image.open(fileName)
                histData = im.histogram()
                for ind,rgbintensityData in enumerate(histData):
                    rgbIndex = ind/256
                    rgbshade = (ind%256)/16
                    imgFeatureList[rgbIndex*16+rgbshade] += rgbintensityData
                if 'headshot' in fileName.split('_')[0]:
                    imgFeatureList.append("headshot")
                elif 'landscape' in fileName.split('_')[0]:
                    imgFeatureList.append("landscape")
                else:
                    imgFeatureList.append("car")
                featureList.append(imgFeatureList)
        return featureList
    
    
    def getLabel(self):
        folder = "carsFolder"
        imageList = []
        imageList.extend([os.path.join(folder,x) for x in os.listdir(folder)])
        featureList = self.loadFeaturesFromList(imageList)
#         print featureList
        self._agent.setFeatureLists(featureList)
        label = self._agent.TestUserInput()
        print "The Image is " + label
    def crossValidate(self,path):
        imageList = []
        for folder in path:
            imageList.extend([os.path.join(folder,x) for x in os.listdir(folder)])
        shuffle(imageList)
        numberImageList = len(imageList)/3
        imageList1 = imageList[0:numberImageList]
        imageList2 = imageList[numberImageList:2*numberImageList]
        imageList3 = imageList[2*numberImageList:3*numberImageList]
        datasets = []
        datasets.append(imageList1)
        datasets.append(imageList2)
        datasets.append(imageList3)
        foldKvalueAccuracyDict = {} 
        for i in range(3):
            foldKvalueAccuracyDict[i] = {}
            for kvalue in [1,3,5,7,9,13,15]:
                foldKvalueAccuracyDict[i][kvalue] = {"correct":0,"truePositive":0,"trueNegative":0,"falsePositive":0,"falseNegative":0}
            trainset1 = trainset2 = testset = []
            trainset1 = list(datasets[i])
            trainset2 = list(datasets[(i+1)%3])
            testset = list(datasets[(i+2)%3])
            trainset1.extend(trainset2)
            trainingFeatureSet = self.loadFeaturesFromList(trainset1)
            testFeatureSet = self.loadFeaturesFromList(testset)
            self._agent.setFeatureLists(trainingFeatureSet)
            for testFeatureList in testFeatureSet:
                distanceList = self._agent.findDistance(testFeatureList)
                for kvalue in [1,3,5,7,9,13,15]:
                    foldKvalueAccuracyDict[i]
                    correct = total = 0
                    label = self._agent.findClass(distanceList, kvalue)
                    if label == "car" and testFeatureList[-1] == "car":
                        foldKvalueAccuracyDict[i][kvalue]["truePositive"] += 1
                    elif label == "car" and testFeatureList[-1] != "car":
                        foldKvalueAccuracyDict[i][kvalue]["falsePositive"] += 1
                    elif label != "car" and testFeatureList[-1] != "car":
                        foldKvalueAccuracyDict[i][kvalue]["trueNegative"] += 1
                    elif label != "car" and testFeatureList[-1] == "car":
                        foldKvalueAccuracyDict[i][kvalue]["falseNegative"] += 1
            for kvalue in [1,3,5,7,9,13,15]:
                foldKvalueAccuracyDict[i][kvalue]["total"] = len(testFeatureSet)
                foldKvalueAccuracyDict[i][kvalue]["accuracy"] = round((float(foldKvalueAccuracyDict[i][kvalue]["truePositive"])+float(foldKvalueAccuracyDict[i][kvalue]["trueNegative"]))/(foldKvalueAccuracyDict[i][kvalue]["truePositive"]+foldKvalueAccuracyDict[i][kvalue]["trueNegative"]+foldKvalueAccuracyDict[i][kvalue]["falsePositive"]+foldKvalueAccuracyDict[i][kvalue]["falseNegative"]),2)
                foldKvalueAccuracyDict[i][kvalue]["sensitivity"] = round((float(foldKvalueAccuracyDict[i][kvalue]["truePositive"]))/(foldKvalueAccuracyDict[i][kvalue]["truePositive"]+foldKvalueAccuracyDict[i][kvalue]["falseNegative"]),2)
                foldKvalueAccuracyDict[i][kvalue]["specificity"] = round((float(foldKvalueAccuracyDict[i][kvalue]["trueNegative"]))/(foldKvalueAccuracyDict[i][kvalue]["trueNegative"]+foldKvalueAccuracyDict[i][kvalue]["falsePositive"]),2)
        return foldKvalueAccuracyDict

if __name__=="__main__":
    environmentObject = Environment()
    environmentObject.getLabel()
#     result = environmentObject.crossValidate(["carsFolder"])