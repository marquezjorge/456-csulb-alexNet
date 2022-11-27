
import os
import shutil


dataSet = "C:\\Users\\Jorge M\\Documents\\longbeach\\456\\dataset"
camoDir = "C:\\Users\\Jorge M\\Documents\\longbeach\\456\\dataset\\Camouflaged"
nonCamoDir = "C:\\Users\Jorge M\Documents\longbeach\\456\\dataset\\NonCamouflaged"
backgroundDir = "C:\\Users\\Jorge M\\Documents\\longbeach\\456\\dataset\\Background"

trainDir = "C:\\Users\\Jorge M\\Documents\\longbeach\\456\\dataset\\train"
testDir = "C:\\Users\\Jorge M\\Documents\\longbeach\\456\dataset\\test"
validateDir = "C:\\Users\\Jorge M\\Documents\\longbeach\\456\\dataset\\validate"


def makeDirectories():
    os.mkdir(trainDir)
    os.mkdir(trainDir + "\\camo")
    os.mkdir(trainDir + "\\nonCamo")
    os.mkdir(trainDir + "\\background")

    os.mkdir(testDir)
    os.mkdir(testDir + "\\camo")
    os.mkdir(testDir + "\\nonCamo")
    os.mkdir(testDir + "\\background")

    os.mkdir(validateDir)
    os.mkdir(validateDir + "\\camo")
    os.mkdir(validateDir + "\\nonCamo")
    os.mkdir(validateDir + "\\background")


# 10% valid, 10% test, 80% train
def moveImages():
    camoImages = [img for img in os.listdir(camoDir)]
    nonCamoImages = [img for img in os.listdir(camoDir)]
    backgroundImages = [img for img in os.listdir(backgroundDir)]
    for i in range(0, 1250):
        # 20% for test
        if i % 5 == 0:
            shutil.move(camoDir + "\\" + camoImages[i], testDir + "\\camo")
            shutil.move(nonCamoDir + "\\" + nonCamoImages[i], testDir + "\\nonCamo")
            shutil.move(backgroundDir + "\\" + backgroundImages[i], testDir + "\\background")
        # 20% for validate
        elif i % 5 == 1:
            shutil.move(camoDir + "\\" + camoImages[i], validateDir + "\\camo")
            shutil.move(nonCamoDir + "\\" + nonCamoImages[i], validateDir + "\\nonCamo")
            shutil.move(backgroundDir + "\\" + backgroundImages[i], validateDir + "\\background")
        # 60% for train
        else:
            shutil.move(camoDir + "\\" + camoImages[i], trainDir + "\\camo")
            shutil.move(nonCamoDir + "\\" + nonCamoImages[i], trainDir + "\\nonCamo")
            shutil.move(backgroundDir + "\\" + backgroundImages[i], trainDir + "\\background")


def reArrangeDataSets():
    dirs = [validateDir, testDir]
    l = ['\\background', "\\camo", "\\nonCamo"]

    # move images from validate/test to train, goal is to increase accuracy
    index = 0
    for i in range(0, len(dirs)):
        for j in range(0, len(l)):
            for img in os.listdir(dirs[i] + l[j]):
                if index % 2 == 0:
                    shutil.move(dirs[i] + l[j] + "\\" + img, trainDir + l[j])
                index += 1
            print(dirs[i] + l[j], index)
            index = 0

    # list number of images in each directory
    for i in range(0, len(dirs)):
        for j in range(0, len(l)):
            print(len(os.listdir(dirs[i] + l[j])))
