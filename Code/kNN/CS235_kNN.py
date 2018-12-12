# Beat tracking example
from __future__ import print_function
import librosa
import librosa.display
import matplotlib
import numpy as np
from numpy.linalg import norm
from dtw import dtw
from array import array
from matplotlib.pyplot import *
import random
import math


def featureExtract(name):
    #print("in featureExtract"+name)
    #================ tempo ================ output: beat per minute
    # Load the audio as a waveform `y`, Store the sampling rate as `sr`
    y, sr = librosa.load(name)
    # Run the default beat tracker
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    #print('Estimated tempo: {:.10f} beats per minute'.format(tempo))
    # Convert the frame indices of beat events into timestamps
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    #================ Freq ================
    '''
    freqs = librosa.cqt_frequencies(24, 55, tuning=0.25)
    librosa.pitch_tuning(freqs)
    print(freqs)
    '''
    #================ Pitch ================
    #y, sr = librosa.load(filename)
    pitches, magnitudes = librosa.core.piptrack(y, sr)
    # Select out pitches with high energy
    pitches = pitches[magnitudes > np.median(magnitudes)]
    librosa.pitch_tuning(pitches)
    #print('Pitch')
    pitches = reduce(lambda x, y: x + y, pitches) / len(pitches)
    #print(pitches)
    #================ Tune ================ output: [-0.5:0.5]
    #y, sr = librosa.load(filename)
    tune = librosa.estimate_tuning(y=y, sr=sr)
    #print('Tune: {:.10f}'.format(tune)) 
    
    #================ CQT ================
    y, sr = librosa.load(name,offset=10, duration=15)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    #print('CQT')
    #print(chroma_cq) 
    
    #================ MFCC ================
    #S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    #mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S))
    mfcc = librosa.feature.mfcc(y, sr)
    #print('MFCC')
    #print(mfcc) 
    
    return tempo, tune, pitches, chroma_cq, mfcc
    #return tempo, tune, pitches

def labelMapping(index):
    inputFolder = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock' ]
    seq = index%100
    #seq = ["%04d" % index%100]
    return inputFolder[index/100], str(seq).zfill(5)
    
def knn(targetinex, trainingSet):
    # there are beat_times, tune, pitches, chroma_cq, mfcc. 5 attributes
    # prepare min/max to normalize them
    minAttribute = [sys.maxsize,sys.maxsize,sys.maxsize,sys.maxsize,sys.maxsize]
    maxAttribute = [-sys.maxsize -1,-sys.maxsize -1,-sys.maxsize -1,-sys.maxsize -1,-sys.maxsize -1]
    label,seq = labelMapping(targetinex)
    filename = 'genres/'+label+'/'+label+'.'+seq+'.au'
    #beat_times, tune, pitches = featureExtract(filename)
    beat_times, tune, pitches, chroma_cq, mfcc = featureExtract(filename)
    
    dataEntityTable = []
    
    for i in trainingSet:
        label,seq = labelMapping(i)
        filename = 'genres/'+label+'/'+label+'.'+seq+'.au'
        print("\t"+filename)
        beat_timesTRAIN, tuneTRAIN, pitchesTRAIN, chroma_cqTRAIN, mfccTRAIN = featureExtract(filename)
        #beat_timesTRAIN, tuneTRAIN, pitchesTRAIN = featureExtract(filename)
        
        dist, cost, acc_cost, path = dtw(chroma_cq.T, chroma_cqTRAIN.T, dist=lambda x, y: norm(x - y, ord=1))
        distCQT = dist
        dist, cost, acc_cost, path = dtw(mfcc.T, mfccTRAIN.T, dist=lambda x, y: norm(x - y, ord=1))
        distMFCC = dist
        dataEntity = [math.fabs(beat_times-beat_timesTRAIN), math.fabs(tune-tuneTRAIN), math.fabs(pitches-pitchesTRAIN), math.fabs(distCQT), math.fabs(distMFCC), i, 0]
        # print(dataEntity)
        # maintain min/max
        if(math.fabs(beat_times-beat_timesTRAIN) >= maxAttribute[0]):
            maxAttribute[0] = math.fabs(beat_times-beat_timesTRAIN)
        if(math.fabs(tune-tuneTRAIN) >= maxAttribute[1]):
            maxAttribute[1] = math.fabs(tune-tuneTRAIN)
        if(math.fabs(pitches-pitchesTRAIN) >= maxAttribute[2]):
            maxAttribute[2] = math.fabs(pitches-pitchesTRAIN)
        if(math.fabs(distCQT) >= maxAttribute[3]):
            maxAttribute[3] = math.fabs(distCQT)
        if(math.fabs(distMFCC) >= maxAttribute[3]):
            maxAttribute[3] = math.fabs(distMFCC)
            
        if(math.fabs(beat_times-beat_timesTRAIN) <= minAttribute[0]):
            minAttribute[0] = math.fabs(beat_times-beat_timesTRAIN)
        if(math.fabs(tune-tuneTRAIN) <= minAttribute[1]):
            minAttribute[1] = math.fabs(tune-tuneTRAIN)
        if(math.fabs(pitches-pitchesTRAIN) <= minAttribute[2]):
            minAttribute[2] = math.fabs(pitches-pitchesTRAIN)
        if(math.fabs(distCQT) <= minAttribute[3]):
            minAttribute[3] = math.fabs(distCQT)
        if(math.fabs(distMFCC) <= minAttribute[3]):
           minAttribute[3] = math.fabs(distMFCC)
        #
        dataEntityTable.append(dataEntity)
    for i,entity in enumerate(dataEntityTable):
        for j in range(5):
            dataEntityTable[i][j] = (dataEntityTable[i][j]-minAttribute[j])/(maxAttribute[j]-minAttribute[j])
        dataEntityTable[i][4] = math.sqrt( dataEntityTable[i][0]*dataEntityTable[i][0] + dataEntityTable[i][1]*dataEntityTable[i][1]\
                       + dataEntityTable[i][2]*dataEntityTable[i][2] + dataEntityTable[i][3]*dataEntityTable[i][3]\
                       + dataEntityTable[i][4]*dataEntityTable[i][4])
    majorVote = sorted(range(len(dataEntityTable)), key=lambda k: dataEntityTable[k][6])
    VoteResult = {'blues':0, 'classical':0, 'country':0, 'disco':0, 'hiphop':0, 'jazz':0, 'metal':0, 'pop':0, 'reggae':0, 'rock':0}
    # use 10 nearest neighbor to major vote
    k = 50
    for i in range(k):
        labelRET,seqRET = labelMapping(dataEntityTable[majorVote[i]][5])
        VoteResult[labelRET] += 1
    labelRET = ""
    countRET = 0
    for key, value in VoteResult.items():
        if value >= countRET:
            countRET = value
            labelRET = key
    print(labelRET)
    return labelRET


#inputFolder = [0,1,2,3,4,5,6,7,8,9]
inputFolder = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock' ]
# total number is 1000, 100 for each category
# use 10 fold, 900
for i in range(1):
    Order = random.sample(range(1000), 1000)
    print(len(Order))
    print(min(Order))
    print(max(Order))
    testing = Order[0:100]  #Order[0:100]
    training = Order[100:1000]  #Order[100:1000]
    
    recallTP = {'blues':0, 'classical':0, 'country':0, 'disco':0, 'hiphop':0, 'jazz':0, 'metal':0, 'pop':0, 'reggae':0, 'rock':0}
    recallALL = {'blues':0, 'classical':0, 'country':0, 'disco':0, 'hiphop':0, 'jazz':0, 'metal':0, 'pop':0, 'reggae':0, 'rock':0}
    precisionTP = {'blues':0, 'classical':0, 'country':0, 'disco':0, 'hiphop':0, 'jazz':0, 'metal':0, 'pop':0, 'reggae':0, 'rock':0}
    precisionALL = {'blues':0, 'classical':0, 'country':0, 'disco':0, 'hiphop':0, 'jazz':0, 'metal':0, 'pop':0, 'reggae':0, 'rock':0}
    
    success = 0
    for j in testing:
        label,seq = labelMapping(j)
        filename = 'genres/'+label+'/'+label+'.'+seq+'.au'
        #print(label+"_"+str(seq))
        print(filename)
        predictLabel = knn(j, training)
        precisionALL[predictLabel] += 1
        #print(predictLabel)
        if predictLabel == label:
            recallTP[predictLabel] += 1
            recallALL[predictLabel] += 1
            precisionTP[predictLabel] += 1
            success += 1
        else:
            recallALL[label] += 1
    
    recall = []
    for key, value in recallALL.items():
        if value != 0:
            recall.append(recallTP[key]/value) 
    precision = []
    for key, value in precisionALL.items():
        if value != 0:
            precision.append(precisionTP[key]/value) 
            
    print("accuracy: "+str(success)+"/10")
    print("recall:"+str(sum(recall, 0.0) / len(recall)))
    print(recallTP)
    print(recallALL)
    print("presicsion:"+str(sum(precision, 0.0) / len(precision)))
    print(recallTP)
    print(recallALL)