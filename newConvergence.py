import sys
import random
import statistics as stat
import math
from numba.typed import List
from bitstring import Bits, BitStream, BitArray, ConstBitStream
from setup import readInput
import os

model, nSite, subSpace, mlStart, ciPow, trainSampleSize, testSampleSize, newGenerationSize, gibStepTrain, gibStepGen, ciThresh, hidNode, trainBatchSize, lr, epoch, nStates, s2Target, maxItr, startSpinTargetItr, energyTola, spinTola, beta, jVal, det, Ms,  posibleDet, bondOrder, outputfile, restart, saveBasis = readInput()

def updateDeterminatList(allDet, allCi, newGen, ci):

    allCi = list(allCi)

    similar_idx = []

    for idx, elem in enumerate(allDet):

        if elem in newGen:
            similar_idx.append(idx)

    for idx1, idx2 in enumerate(similar_idx):

        allDet.pop(idx2-idx1)
        allCi.pop(idx2-idx1)

    for idx, elem in enumerate(newGen):

        allDet.append(elem)
        allCi.append(ci[idx])

    return allDet, allCi

def makeFitGeneration(basis, ci):

    ciOrdered = sorted(abs(ci), reverse=True)

    fitness=[]
    ciFit = []

    for x in ciOrdered:

        ix = list(abs(ci)).index(x)

        fitness.append(basis[ix])
        ciFit.append(ci[ix])

    return fitness, ciFit

def makeComplementaryGeneration(basis, ci):

    fitness=[]
    ciFit = []

    for x in ci:

        ix = list(ci).index(x)

        fitness.append(basis[ix])
        ciFit.append(ci[ix])

        if (Ms == 0) and ~basis[ix] not in fitness:

            fitness.append(~basis[ix])
            ciFit.append(ci[ix])

    return fitness, ciFit

def convInitializer():

    targetState = [100, 101]

    s2ValDiff = [0.0, 0.0]

    energyChange = [1.0, 1.0, 1.0, 1.0, 1.0]
    spinChange = [10.0, 10.0, 10.0, 10.0, 10.0]

    s2ValList = List()

    [s2ValList.append(0.0) for x in range(nStates)]

    return targetState, s2ValList, s2ValDiff, energyChange, spinChange

def update(energy, basis, ciCoef):

    energySave = energy
    ciSave = ciCoef
    basisSave = basis

    return energySave, basisSave, ciSave

def checkConvergence(eMin, eNew, s2Min, s2New, allDet, ciCoefNew, targetState, s2ValDiff, itr):

    Eith = eMin

    if ((s2ValDiff[1] - s2ValDiff[0] <= spinTola) or ((eNew <= eMin) or (random.random() < math.exp(-(beta * (eNew - eMin)))))):

        eMin = eNew
        s2Min = s2New

        s2ValDiff[0] = s2ValDiff[1]

        targetState[0] = targetState[1]

        allDet = allDet
        allCicoef = ciCoefNew

        energyUpdate = True

    else:

        allDet = allDet
        allCicoef = ciCoefNew

        energyUpdate = False

    newline = ("ite->\t%d ; spece->\t%d ; Energy->\t%f ; State->\t%d ; s^2 Expe Val->\t%2.4f ;\n")%((itr +1), len(allDet), round(eMin, 6), targetState[0] + 1, round(s2Min,4))

    with open(outputfile, "a") as fout:
        fout.write(newline)

    return allDet, allCicoef, eMin, s2ValDiff, s2Min, energyUpdate

def checkFinalConv(energyChange, spinChange, convReach):

    if (abs(energyChange) < energyTola) and (abs(spinChange) < spinTola):

        convReach = True

    return convReach



