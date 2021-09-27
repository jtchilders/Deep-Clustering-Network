import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import pdb
import collections
import globals

def getPreselections():
    global selections
    selections = collections.OrderedDict()

    selections["1"]="(1)"
    selections["presel_common"]="(passTightCleanDFFlag==1)*(nj_good>=2)*((tcMeTCategory==1)||(tcMeTCategory<0))*(pT_1jet>250)*(num_bjets==0)*((GenFiltMET<100 )|| (RunNumber!=410470))"

    selections["presel_0lep"]=selections['presel_common']+"*(METTrigPassed)*(nsignalLep==0)*(nbaselineLep==0)*(eT_miss>250)*(minDPhi_4jetMET>0.4)"
    
    selections["presel_0lep_1cjet20"]=selections['presel_0lep']+"*(num_cjets20>=1)"
    selections["presel_0lep_1cjet30"]=selections['presel_0lep']+"*(num_cjets30>=1)"
    selections["presel_0lep_1cjet40"]=selections['presel_0lep']+"*(num_cjets40>=1)"
    selections["presel_0lep_2cjet20"]=selections['presel_0lep']+"*(num_cjets20>=2)"
    selections["presel_0lep_2cjet30"]=selections['presel_0lep']+"*(num_cjets30>=2)"
    selections["presel_0lep_2cjet40"]=selections['presel_0lep']+"*(num_cjets40>=2)"

    selections["SRA"]=selections['presel_0lep']+"*(num_cjets20>=2)*(MTcMin20>100)*(m_cc20>150)*(metsigST>5)"

    selections["VRW"] = selections['presel_0lep_2cjet20']+'*(m_cc20>150)*(MTcMin20<100)'
    selections["VRZ"] = selections['presel_0lep_2cjet20']+'*(m_cc20<150)*(MTcMin20>100)'
    
    selections["presel_1lep"]=selections['presel_common']+"*(METTrigPassed)*(nsignalLep==1)*(nbaselineLep==1)*(eT_miss>250)*(minDPhi_4jetMET>0.4)*(pT_1lep>20)"

    selections["presel_2lep"]=selections['presel_common']+"*(nsignalLep==2)*(nbaselineLep==2)*(mll>81)*(mll<101)*(eT_miss<200)*(eT_miss_prime>250)*(minDPhi_4jetMET_prime>0.4)"

    selections["presel_ML"]="(passTightCleanDFFlag==1)*(nj_good>=2)*((tcMeTCategory==1)||(tcMeTCategory<0))*(pT_1jet>250)*(METTrigPassed)*(nsignalLep==0)*(nbaselineLep==0)*(eT_miss>200)"


def get_DataLoaders(args: argparse.Namespace, signals: list, trainBranches: list):

    # Load data
    allSamps = pd.read_hdf(args.data)

    # get selection
    getPreselections()
    presel = selections['presel_ML'].replace('*', ' & ').replace('||', ' | ').replace('(', '(allSamps.').replace('allSamps.(', '(')
    print(presel)

    # apply selection
    rawDataPresel = allSamps[eval(presel)].copy()
    sampSizes = {}
    sampYields = {}
    print('sampNames: ', sorted(pd.unique(rawDataPresel.sampName)))
    for sampName in sorted(pd.unique(rawDataPresel.sampName)):
        sampIndex = rawDataPresel['sampName']==sampName
        sampYield = rawDataPresel[sampIndex].weight.sum()
        if sampYield == 0:
            print("Dropping", sampName)
            rawDataPresel.drop(sampIndex)
        sampSizes[sampName] = rawDataPresel[sampIndex].shape[0]
        sampYields[sampName] = sampYield
        # print(sampName, round(sampYield,1), sampIndex.sum())

    # calculate weight to balance signal samples
    maxYieldKey = max(sampYields, key=lambda k: sampYields[k])
    tempDFs = []
    goodSamps = pd.unique(rawDataPresel.sampName)
    for sampName in sampYields:
        sampIndex = rawDataPresel['sampName']==sampName
        rawDataPresel.loc[sampIndex, 'clus_weight'] = rawDataPresel.loc[sampIndex, 'weight']*(sampYields[maxYieldKey]/sampYields[sampName])
        #print(rawDataPresel[sampIndex].clus_weight.sum())

    # columns to be used, get rid of 'sampName'
    inputColumns = trainBranches.copy()
    inputColumns.remove('sampName')

    # scale data: normalize to max value
    scaledData = rawDataPresel.copy(deep=True)
    scalers = {}
    maxvalue = 0.
    for column in trainBranches:
        if 'sampName' in column: continue
        # scalers[column] = MinMaxScaler()
        # scaledData[[column]] = scalers[column].fit_transform(scaledData[[column]])
        colmax = scaledData[column].max()
        if colmax > maxvalue:
            maxvalue = colmax
    scaledData[inputColumns] = scaledData[inputColumns] / maxvalue

    # select signal samples
    print('using only samples: ', signals)
    signalsMap = {}
    for i, signal in enumerate(signals):
        signalsMap[signal] = i
        try:
            signalMask |= scaledData.sampName == signal
        except:
            signalMask = scaledData.sampName == signal
    twoSigs = scaledData[signalMask]
    # data for AE
    aeData = twoSigs[trainBranches]
    aeSampName = twoSigs['sampName']

    # I want to make the two samples balanced (similar as above...)
    # I'll calculate the total sum-of-weights for each sample and I'll scale with 1 over it
    norm_weights = {}
    for s in signals:
        sumOfWeights = twoSigs[twoSigs.sampName == s].AnalysisWeight.sum()
        norm_weights[s] = sumOfWeights

    # add to new column called norm_weight
    for s in signals:
        norm_weight = norm_weights[s]
        sampIndex = twoSigs.sampName == s
        twoSigs.loc[sampIndex, 'norm_weight'] = twoSigs.loc[sampIndex, 'AnalysisWeight']/(norm_weight)

    # select which weight to be used. In the current implementation of DCN this is not used anyways
    # tempWeights = twoSigs['AnalysisWeight']
    # tempWeights = twoSigs['clus_weight']
    tempWeights = twoSigs['norm_weight']

    # print data characteristics
    print('aeData.shape:', aeData.shape)
    print('aeData.columns:', aeData.columns) 
    print('tempWeights.shape:', tempWeights.shape)
    print('samples:', aeSampName.unique())
    tmp = aeData.apply(lambda s: pd.Series([s.min(), s.max()],index=['min', 'max']))
    print(f'tmp = {tmp}')

    # batch size and test/train fraction
    batch_size = args.batch_size
    testTrainFrac = 0.7
    # config for throughput timing
    # curently vaiable with the available data. 
    # Vangelis: setting when using all the signal samples
    if args.time_throughput:
        batch_size = globals.timing_batch_size = 2**19
        testTrainFrac = 0.9
        args.epoch = globals.timing_epochs = 100
        
    # select events
    nEvents=aeData.shape[0]
    nTrain = np.floor(nEvents*testTrainFrac)
    # use an integer number of batches, i.e. all batches used have the same number of events
    newNTrain = int(np.floor(nTrain/batch_size)*batch_size)
    msk = np.random.choice(nEvents, newNTrain, replace=False)
    
    # train np.arrays
    train_inputs = aeData[inputColumns].to_numpy()[msk]
    train_targets = aeData['sampName'].map(signalsMap).to_numpy()[msk]

    # test np.arrays
    test_inputs = aeData[inputColumns].to_numpy()[~msk]
    test_targets = aeData['sampName'].map(signalsMap).to_numpy()[~msk]

    # train & test weights. not used anyways
    trainWeights = torch.Tensor(tempWeights.to_numpy()[msk])
    testWeights = torch.Tensor(tempWeights.to_numpy()[~msk])

    # torch TensorDatasets
    trainDataset = torch.utils.data.TensorDataset(torch.Tensor(train_inputs), torch.Tensor(train_targets), trainWeights) # create your datset
    testDataset = torch.utils.data.TensorDataset(torch.Tensor(test_inputs), torch.Tensor(test_targets), testWeights) # create your datset
    
    # torch DataLoaders
    train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size)

    return train_loader, test_loader