import time
import os
import sys
from shutil import copyfile
import csv
import math
from math import factorial as fact
from bitstring import BitArray 

inputfile = sys.argv[1]

def readInput():

    
    outputfile = str(inputfile) + '.out'
    
    copyfile(inputfile, outputfile)

    fin = open(inputfile,"r")
    lines = fin.readlines()
    length = len(lines)
    restart = False
    saveBasis = 'nothing.dat'
    for i in range(length):
        toks = lines[i].split(",")
        if len(toks) >= 2:

            if toks[0] == 'model':
                model = toks[1].strip()

            if toks[0] == 'nSite':
                nSite = int(toks[1])
            
            if toks[0] == 'subSpace':
                subSpace = int(toks[1])
            
            if toks[0] == 'mlStart':
                mlStart = int(toks[1])

            if toks[0] == 'ciPow':
                ciPow = float(toks[1])

            if toks[0] == 'trainSampleSize':
                trainSampleSize = int(toks[1])

            if toks[0] == 'testSampleSize':
                testSampleSize = int(toks[1])

            if toks[0] == 'newGenerationSize':
                newGenerationSize = int(toks[1])

            if toks[0] == 'gibStepTrain':
                gibStepTrain = int(toks[1])

            if toks[0] == 'gibStepGen':
                gibStepGen = int(toks[1])

            if toks[0] == 'ciThresh':
                ciThresh = float(toks[1])
            
            if toks[0] == 'hidNode':
                hidNode = int(toks[1])

            if toks[0] == 'trainBatchSize':
                trainBatchSize = int(toks[1])

            if toks[0] == 'lr':
                lr = float(toks[1])
            if toks[0] == 'epoch':
                epoch = int(toks[1])

            if toks[0] == 'nStates':
                nStates = int(toks[1])

            if toks[0] == 's2Target':
                s2Target = float(toks[1])

            if toks[0] == 'maxItr':
                maxItr = int(toks[1])

            if toks[0] == 'startSpinTargetItr':
                startSpinTargetItr = int(toks[1])

            if toks[0] == 'energyTola':
                energyTola = float(toks[1])

            if toks[0] == 'spinTola':
                spinTola = float(toks[1])

            if toks[0] == 'beta':
                beta = float(toks[1])
            
            if toks[0] == 'bondOrder':
                bondOrder = str(toks[1]).strip()

            if toks[0] == 'jValue':
                jVal = -float(toks[1])

            if toks[0] == 'restart':
                if toks[1] == 'True':
                    restart = True
                    saveBasis = str(toks[2]).strip()

            
            if toks[0] == 'Ms':
                noOfMs = int(toks[1])
                Ms = int( toks[2]) 
                up = int((nSite/2)  + Ms)
                down = nSite - up
                posibleDet = int(fact(nSite)/(fact(nSite - up) * fact(up)))
                if up == down:
                    tem = [0, 1] * down
                    det = BitArray(tem)
                elif up - down == 2:
                    tem = tem = [0, 1] * down
                    det = BitArray(tem)
                    det = det + '0b1' + '0b1'         
                        
    return model, nSite, subSpace, mlStart, ciPow, trainSampleSize, testSampleSize, newGenerationSize, gibStepTrain, gibStepGen, ciThresh, hidNode, trainBatchSize, lr, epoch, nStates, s2Target, maxItr, startSpinTargetItr, energyTola, spinTola, beta, jVal, det, Ms,  posibleDet, bondOrder, outputfile, restart, saveBasis 


