import time
import os
import sys
from shutil import copyfile
from setup import readInput
from MCCI_thresh import performMCCI
import warnings
warnings.filterwarnings("ignore")


start = time.time()

model, nSite, subSpace, mlStart, ciPow, trainSampleSize, testSampleSize, newGenerationSize, gibStepTrain, gibStepGen, ciThresh, hidNode, trainBatchSize, lr, epoch, nStates, s2Target, maxItr, startSpinTargetItr, energyTola, spinTola, beta, jVal, det, Ms,  posibleDet, bondOrder, outputfile, restart, saveBasis = readInput()

newline = ("\nTotal Posible Determinats are %d .\nBreakup are [Ms, No of Determinants] - ")% posibleDet
with open (outputfile, "a") as fout:
    fout.write(newline)

newline = ("\t[%d, %d]")%(Ms, posibleDet) 
with open(outputfile, "a") as fout:
    fout.write(newline)
    fout.write("\n\n")

if (subSpace > posibleDet * 0.8):
    sys.exit("Sub-Space size is more than 80 % of total determinants space. Make Sub-Space size smaller and run it again.\n ")

performMCCI()

newline = ("Total Time Taken in MCCI Calculation is %f sec.")%( time.time() - start )
with open(outputfile, "a") as fout:
    fout.write(newline)
