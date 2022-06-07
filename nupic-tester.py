from components.encoders import WordEncoder 
from components.sdr import *
from nupic.pool import SpatialPooler as SP
from nupic.temporal_memory import TemporalMemory
from components.scalars import Encoder
import numpy as np
from tqdm import tqdm
import matplotlib
import random
import matplotlib.pyplot as plt
import seaborn as sns

def heatMap(input):
    """
    Accepts numpy array, displays heatmap
    """
    uniform_data = np.random.rand(10, 12)
    ax = sns.heatmap(uniform_data, linewidth=0.5)
    plt.show()

def percentOverlap(x1, x2, size):
  """
  Computes the percentage of overlap between vectors x1 and x2.

  @param x1   (array) binary vector
  @param x2   (array) binary vector
  @param size (int)   length of binary vectors

  @return percentOverlap (float) percentage overlap between x1 and x2
  """
  nonZeroX1 = np.count_nonzero(x1)
  nonZeroX2 = np.count_nonzero(x2)
  minX1X2 = min(nonZeroX1, nonZeroX2)
  percentOverlap = 0
  if minX1X2 > 0:
    percentOverlap = float(np.dot(x1, x2))/float(minX1X2)
  return percentOverlap

def corruptVector(vector, noiseLevel):
  """
  Corrupts a binary vector by inverting noiseLevel percent of its bits.

  @param vector     (array) binary vector to be corrupted
  @param noiseLevel (float) amount of noise to be applied on the vector.
  """
  size = len(vector)
  for i in range(size):
    rnd = random.random()
    if rnd < noiseLevel:
      if vector[i] == 1:
        vector[i] = 0
      else:
        vector[i] = 1

def resetVector(x1, x2):
  """
  Copies the contents of vector x1 into vector x2.

  @param x1 (array) binary vector to be copied
  @param x2 (array) binary vector where x1 is copied
  """
  size = len(x1)
  for i in range(size):
    x2[i] = x1[i]

def sequence(sentence):
        w = WordEncoder()  
        sequence = [] 
        for i in tqdm(range(len(sentence))):
            sequence.append(dimensionalityReduction(w.encode(sentence[i])))
        return sequence
        
def Test1():
    random.seed(1)
    uintType = "uint32"
    inputDimensions = (32,32)
    columnDimensions = (64,64)
    inputSize = np.array(inputDimensions).prod()
    columnNumber = np.array(columnDimensions).prod()
    inputArray = np.zeros(inputSize, dtype=uintType)

    for i in range(inputSize):
        inputArray[i] = random.randrange(2)

    print(inputArray)
    activeCols = np.zeros(columnNumber, dtype=uintType)

    sp = SP(
            input_dims=inputDimensions,
            minicolumn_dims=columnDimensions,
            active_minicolumns_per_inh_area=10,
            local_density=-1.0,
            potential_radius=int(0.5*inputSize),
            potential_percent=0.5,
            global_inhibition=False,
            stimulus_threshold=0,
            synapse_perm_inc=0.03,
            synapse_perm_dec=0.008,
            synapse_perm_connected=0.1,
            min_percent_overlap_duty_cycles=0.001,
            duty_cycle_period=1000,
            boost_strength=0.0,
            seed=1,
        )
    # Part 1:
    # -------
    # A column connects to a subset of the input vector (specified
    # by both the potentialRadius and potentialPct). The overlap score
    # for a column is the number of connections to the input that become
    # active when presented with a vector. When learning is 'on' in the SP,
    # the active connections are reinforced, whereas those inactive are
    # depressed (according to parameters synPermActiveInc and synPermInactiveDec.
    # In order for the SP to create a sparse representation of the input, it
    # will select a small percentage (usually 2%) of its most active columns,
    # ie. columns with the largest overlap score.
    # In this first part, we will create a histogram showing the overlap scores
    # of the Spatial Pooler (SP) after feeding it with a random binary
    # input. As well, the histogram will show the scores of those columns
    # that are chosen to build the sparse representation of the input.

    sp.compute(inputArray, False, activeCols)
    overlaps = sp.calculate_overlap(inputArray)
    activeColsScores = []
    for i in activeCols.nonzero():
        activeColsScores.append(overlaps[i])

    bins = np.linspace(min(overlaps), max(overlaps), 28)
    plt.hist(overlaps, bins, alpha=0.5, label="All cols")
    plt.hist(activeColsScores, bins, alpha=0.5, label="Active cols")
    plt.legend(loc="upper right")
    plt.xlabel("Overlap scores")
    plt.ylabel("Frequency")
    plt.title("Figure 1: Column overlap of a SP with random input.")
    plt.savefig("figure_1")
    plt.close()

    # Part 2a:
    # -------
    # The input overlap between two binary vectors is defined as their dot product.
    # In order to normalize this value we divide by the minimum number of active
    # inputs (in either vector). This means we are considering the sparser vector as
    # reference. Two identical binary vectors will have an input overlap of 1,
    # whereas two completely different vectors (one is the logical NOT of the other)
    # will yield an overlap of 0. In this section we will see how the input overlap
    # of two binary vectors decrease as we add noise to one of them.

    inputX1 = np.zeros(inputSize, dtype=uintType)
    inputX2 = np.zeros(inputSize, dtype=uintType)
    outputX1 = np.zeros(columnNumber, dtype=uintType)
    outputX2 = np.zeros(columnNumber, dtype=uintType)

    for i in range(inputSize):
        inputX1[i] = random.randrange(2)

    x = []
    y = []
    for noiseLevel in np.arange(0, 1.1, 0.1):
        resetVector(inputX1, inputX2)
        corruptVector(inputX2, noiseLevel)
        x.append(noiseLevel)
        y.append(percentOverlap(inputX1, inputX2, inputSize))


    plt.plot(x, y)
    plt.xlabel("Noise level")
    plt.ylabel("Input overlap")
    plt.title("Figure 2: Input overlap between 2 identical vectors in function of "
            "noiseLevel.")
    plt.savefig("figure_2")
    plt.close()

    # Part 2b:
    # -------
    # The output overlap between two binary input vectors is the overlap of the
    # columns that become active once they are fed to the SP. In this part we
    # turn learning off, and observe the output of the SP as we input two binary
    # input vectors with varying level of noise.
    # Starting from two identical vectors (that yield the same active columns)
    # we would expect that as we add noise to one of them their output overlap
    # decreases.
    # In this part we will show how the output overlap behaves in function of the
    # input overlap between two vectors.
    # Even with an untrained spatial pooler, we see some noise resilience.
    # Note that due to the non-linear properties of high dimensional SDRs, overlaps
    # greater than 10 bits, or 25% in this example, are considered significant.

    x = []
    y = []
    for noiseLevel in np.arange(0, 1.1, 0.1):
        resetVector(inputX1, inputX2)
        corruptVector(inputX2, noiseLevel)

        sp.compute(inputX1, False, outputX1)
        sp.compute(inputX2, False, outputX2)

        x.append(percentOverlap(inputX1, inputX2, inputSize))
        y.append(percentOverlap(outputX1, outputX2, columnNumber))


    plt.plot(x, y)
    plt.xlabel("Input overlap")
    plt.ylabel("Output overlap")
    plt.title("Figure 3: Output overlap in function of input overlap in a SP "
            "without training")
    plt.savefig("figure_3")
    plt.close()

    # Part 3:
    # -------
    # After training, a SP can become less sensitive to noise. For this purpose, we
    # train the SP by turning learning on, and by exposing it to a variety of random
    # binary vectors. We will expose the SP to a repetition of input patterns in
    # order to make it learn and distinguish them once learning is over. This will
    # result in robustness to noise in the inputs. In this section we will reproduce
    # the plot in the last section after the SP has learned a series of inputs. Here
    # we will see how the SP exhibits increased resilience to noise after learning.

    # We will present 10 random vectors to the SP, and repeat this 30 times.
    # Later you can try changing the number of times we do this to see how it
    # changes the last plot. Then, you could also modify the number of examples to
    # see how the SP behaves. Is there a relationship between the number of examples
    # and the number of times that we expose them to the SP?

    numExamples = 10
    inputVectors = np.zeros((numExamples, inputSize), dtype=uintType)
    outputColumns = np.zeros((numExamples, columnNumber), dtype=uintType)

    for i in range(numExamples):
        for j in range(inputSize):
            inputVectors[i][j] = random.randrange(2)

    # This is the number of times that we will present the input vectors to the SP
    epochs = 30

    for _ in range(epochs):
        for i in range(numExamples):
            #Feed the examples to the SP
            sp.compute(inputVectors[i][:], True, outputColumns[i][:])

    plt.plot(sorted(overlaps)[::-1], label="Before learning")
    overlaps = sp.calculate_overlap(inputArray)
    plt.plot(sorted(overlaps)[::-1], label="After learning")
    plt.axvspan(0, len(activeColsScores[0]), facecolor="g", alpha=0.3,
                label="Active columns")
    plt.legend(loc="upper right")
    plt.xlabel("Columns")
    plt.ylabel("Overlap scores")
    plt.title("Figure 4a: Sorted column overlaps of a SP with random "
            "input.")
    plt.savefig("figure_4a")
    plt.close()


    inputVectorsCorrupted = np.zeros((numExamples, inputSize), dtype=uintType)
    outputColumnsCorrupted = np.zeros((numExamples, columnNumber), dtype=uintType)

    x = []
    y = []
    # We will repeat the experiment in the last section for only one input vector
    # in the set of input vectors
    for noiseLevel in np.arange(0, 1.1, 0.1):
        resetVector(inputVectors[0][:], inputVectorsCorrupted[0][:])
        corruptVector(inputVectorsCorrupted[0][:], noiseLevel)

        sp.compute(inputVectors[0][:], False, outputColumns[0][:])
        sp.compute(inputVectorsCorrupted[0][:], False, outputColumnsCorrupted[0][:])

        x.append(percentOverlap(inputVectors[0][:], inputVectorsCorrupted[0][:],
                                inputSize))
        y.append(percentOverlap(outputColumns[0][:], outputColumnsCorrupted[0][:],
                                columnNumber))


    plt.plot(x, y)
    plt.xlabel("Input overlap")
    plt.ylabel("Output overlap")
    plt.title("Figure 4: Output overlap in function of input overlap in a SP after "
            "training")
    plt.savefig("figure_4")
    plt.close()


def Test2(epochs):
    size = 128
    print('Starting...')
    sentence = ['green','red','blue']
    seq = sequence(sentence)
    #VISUALIZE ENCODED SEQUENCE
    for i in range(len(seq)):
        saveSDR(to2d(seq[i]),sentence[i])
    random.seed(1)
    uintType = "uint32"
    inputDimensions = (len(seq[0]),1)
    columnDimensions = (size,size)
    inputSize = np.array(inputDimensions).prod()
    columnNumber = np.array(columnDimensions).prod()

    activeCols = np.zeros(columnNumber, dtype=uintType)
    print('Initializing Spatial Pooler')
    sp = SP(
            input_dims=inputDimensions,
            minicolumn_dims=columnDimensions,
            active_minicolumns_per_inh_area=10,
            local_density=-1.0,
            potential_radius=int(0.5*inputSize),
            potential_percent=0.5,
            global_inhibition=True,
            stimulus_threshold=0,
            synapse_perm_inc=0.03,
            synapse_perm_dec=0.008,
            synapse_perm_connected=0.1,
            min_percent_overlap_duty_cycles=0.001,
            duty_cycle_period=1000,
            boost_strength=0.0,
            seed=1,
        )
    
    #LEARNIG A SEQUENCE
    for i in range(epochs):
        for i in range(len(seq)):
            print(f'Learning sequence {i}')
            print('Connnecting synapses to input space')
            active = sp.compute(seq[i], True, activeCols)
            overlaps = sp.get_boosted_overlaps()
            #print('Current Overlaps', heatMap(overlaps))
            print('Vizualizing Potential Pool')
            saveFromActive(size**2,active)


def Test3(epochs):
    print('Starting...')
    seq = []
    #VISUALIZE ENCODED SEQUENCE
    for i in range(5):
        seq.append(generate_sdr(16348,0.02,dimensionality=1))
    random.seed(1)
    uintType = "uint32"
    inputDimensions = (len(seq[0]),1)
    columnDimensions = (64,64)
    inputSize = np.array(inputDimensions).prod()
    columnNumber = np.array(columnDimensions).prod()
    activeCols = np.zeros(columnNumber, dtype=uintType)
    print('Initializing Spatial Pooler')
    sp = SP(
            input_dims=inputDimensions,
            minicolumn_dims=columnDimensions,
            active_minicolumns_per_inh_area=10,
            local_density=-1.0,
            potential_radius=int(0.5*inputSize),
            potential_percent=0.5,
            global_inhibition=True,
            stimulus_threshold=0,
            synapse_perm_inc=0.03,
            synapse_perm_dec=0.008,
            synapse_perm_connected=0.1,
            min_percent_overlap_duty_cycles=0.001,
            duty_cycle_period=1000,
            boost_strength=0.0,
            seed=1,
        )
    tp = TemporalMemory(
               columnDimensions=(columnDimensions,),
               cellsPerColumn=32,
               activationThreshold=13,
               initialPermanence=0.21,
               connectedPermanence=0.50,
               minThreshold=10,
               maxNewSynapseCount=20,
               permanenceIncrement=0.10,
               permanenceDecrement=0.10,
               predictedSegmentDecrement=0.0,
               maxSegmentsPerCell=255,
               maxSynapsesPerSegment=255,
               seed=42)

    #LEARNIG A SEQUENCE
    for i in range(epochs):
        for i in range(len(seq)):
            print(f'Learning sequence {i}')
            print('Connnecting synapses to input space')
            active = sp.compute(seq[i], True, activeCols)
            overlaps = sp.get_boosted_overlaps()
            #print('Current Overlaps', heatMap(overlaps))
            print('Vizualizing Potential Pool')
            tp.compute(active,True)
            predicted = tp.getPredictiveCells()
            print(predicted)

Test3(1)