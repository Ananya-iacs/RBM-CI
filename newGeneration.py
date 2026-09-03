import random
from bitstring import Bits, BitStream, BitArray, ConstBitStream
from setup import readInput
import time
from tqdm import tqdm

model, nSite, subSpace, mlStart, ciPow, trainSampleSize, testSampleSize, newGenerationSize, gibStepTrain, gibStepGen, ciThresh, hidNode, trainBatchSize, lr, epoch, nStates, s2Target, maxItr, startSpinTargetItr, energyTola, spinTola, beta, jVal, det, Ms,  posibleDet, bondOrder, outputfile, restart, saveBasis = readInput()

predictDataFile = outputfile + ".predictData.csv"
enrichDataFile = outputfile + ".enrich.csv"

f1 = open(bondOrder)
bO1=[]
bO2=[]
for line in f1:
    line = line.split()
    bO1.append(int(line[0])-1)
    bO2.append(int(line[1])-1)

orderlength = len(bO1)

zero = BitArray(nSite)
one = zero[1 : nSite] + '0b1'

def mutation (determinantOriginal):
    determinant = determinantOriginal.copy()
    flag = 0

    while(flag == 0):
        i = random.randint(0, nSite-1)
        j = random.randint(0, nSite-1)

        if (determinant[i] != determinant[j]):
            determinant[i] , determinant[j] = determinant[j] , determinant[i]
            flag = 1

    return determinant, ~determinant

def allConnectedDeterminants(determinantOriginal):

    connected = []

    for i in range(orderlength):

        determinant = determinantOriginal.copy()

        if determinant[bO1[i]] != determinant[bO2[i]]:

            determinant[bO1[i]], determinant[bO2[i]] = \
                determinant[bO2[i]], determinant[bO1[i]]

            connected.append(determinant)

    return connected

def reflection (deternminantOriginal) -> int:
    n = deternminantOriginal.copy()
    rev = zero

    for i in range(nSite):
        bit = ( n >> i) & one
        rev = rev | (bit << (nSite -1 -i))

    return rev, ~rev

def mutationiConected (determinantOriginal):

    determinant = determinantOriginal.copy()
    flag = 0

    while(flag == 0):

        i = random.randint(0, orderlength-1)

        if (determinant[bO1[i]] != determinant[bO2[i]]):

            determinant[bO1[i]] , determinant[bO2[i]] = determinant[bO2[i]] , determinant[bO1[i]]
            flag = 1

    return determinant, ~determinant

def makeNewGeneration(subBasis):

    newGen = subBasis.copy()
    lenSB = len(subBasis)

    # FAST LOOKUP
    newGenSet = set()

    for x in newGen:
        newGenSet.add(x.bin)

    target_size = min(subSpace, posibleDet)

    pbar = tqdm(total=(target_size - lenSB), desc="MCCI data generation")

    attempt = 0
    stagnation = 0

    max_attempt = max(500000, 200 * target_size)
    max_stagnation = 100000

    last_len = len(newGen)

    while (len(newGen) < target_size):

        attempt += 1

        if attempt > max_attempt:
            break

        indx = random.randint(0, (len(subBasis) -1))
        prob = random.random()

        basisCopy = (subBasis[indx]).copy()

        added = False

        if (prob >= 0.5):

            mutated, compliMutated = mutation(basisCopy)

            if mutated.bin not in newGenSet:

                 newGen.append(mutated)
                 newGenSet.add(mutated.bin)

                 pbar.update(1)

                 added = True

                 if Ms == 0 and compliMutated.bin not in newGenSet:

                    newGen.append(compliMutated)
                    newGenSet.add(compliMutated.bin)

                    pbar.update(1)

        if (prob < 0.5):

            reflected, compliReflected = reflection( basisCopy)

            if reflected.bin not in newGenSet:

                newGen.append(reflected)
                newGenSet.add(reflected.bin)

                pbar.update(1)

                added = True

                if Ms == 0 and compliReflected.bin not in newGenSet:

                    newGen.append(compliReflected)
                    newGenSet.add(compliReflected.bin)

                    pbar.update(1)

        if len(newGen) == last_len:
            stagnation += 1
        else:
            stagnation = 0
            last_len = len(newGen)

        if stagnation > max_stagnation:
            break

    pbar.close()

    return newGen, len(newGen)



