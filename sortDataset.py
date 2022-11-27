
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


camoImages = [img for img in os.listdir(camoDir)]
nonCamoImages = [img for img in os.listdir(nonCamoDir)]
backgroundImages = [img for img in os.listdir(backgroundDir)]


def moveImages():
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


makeDirectories()
moveImages()
