import torch
import argparse
import numpy as np
import pandas as pd
from DCN import DCN
from sklearn.preprocessing import MinMaxScaler
# from torchvision import datasets, transforms
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score

import fnmatch,collections

selections=collections.OrderedDict()

def getPreselections():

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


getPreselections()

def evaluate(model, test_loader):
    mse = []
    for data, target, weights in test_loader:
        data = data.to(model.device)
        latent_X = model.autoencoder(data, latent=True)
        latent_X = latent_X.detach().cpu().numpy()
        pred_data = model.autoencoder(data)
        pred_data = pred_data.detach().cpu().numpy()

        mse.append(torch.nn.functional.mse_loss(torch.as_tensor(pred_data),torch.as_tensor(data.detach().cpu().numpy())))
        # y_test.append(pred_data.numpy())
        # y_pred.append(model.kmeans.update_assign(latent_X).reshape(-1, 1))


    mse = np.mean(mse)
    return mse


def solver(args, model, train_loader, test_loader):

    rec_loss_list = model.pretrain(train_loader, args.pre_epoch)
    mse_list = []

    for e in range(args.epoch):
        model.train()
        model.fit(e, train_loader)

        model.eval()
        MSE = evaluate(model, test_loader)  # evaluation on test_loader
        mse_list.append(MSE)
        
        print('\nEpoch: {:02d} | MSE: {:.5f} \n'.format(
            e, MSE))

    return rec_loss_list, mse_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Deep Clustering Network')

    # Dataset parameters
    parser.add_argument('--data', default='ccMET_noBackground.h5',
                        help='dataset path')
    parser.add_argument('--input-dim', type=int, default=5,
                        help='input dimension')
    parser.add_argument('--n-classes', type=int, default=10,
                        help='output dimension')

    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='input batch size for training')
    parser.add_argument('--epoch', type=int, default=300,
                        help='number of epochs to train')
    parser.add_argument('--pre-epoch', type=int, default=50, 
                        help='number of pre-train epochs')
    parser.add_argument('--pretrain', type=bool, default=True,
                        help='whether use pre-training')

    # Model parameters
    parser.add_argument('--lamda', type=float, default=1,
                        help='coefficient of the reconstruction loss')
    parser.add_argument('--beta', type=float, default=1,
                        help=('coefficient of the regularization term on '
                              'clustering'))
    parser.add_argument('--hidden-dims', default=[500, 500, 2000],
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--latent_dim', type=int, default=2,
                        help='latent space dimension')
    parser.add_argument('--n-clusters', type=int, default=2,
                        help='number of clusters in the latent space')

    # Utility parameters
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='number of jobs to run in parallel')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='whether to use GPU')
    parser.add_argument('--log-interval', type=int, default=100,
                        help=('how many batches to wait before logging the '
                              'training status'))

    args = parser.parse_args()

    # Load data
    allSamps = pd.read_hdf(args.data)

    presel = selections['presel_0lep_1cjet20'].replace('*', ' & ').replace('||', ' | ').replace('(', '(allSamps.').replace('allSamps.(', '(')
    print(presel)
    #print(sorted(branches))
    myPresel = (allSamps.passTightCleanDFFlag==1) & (allSamps.nj_good>=2) & ((allSamps.tcMeTCategory==1) | (allSamps.tcMeTCategory<0)) & (allSamps.pT_1jet>250) & (allSamps.num_bjets==0) & ((allSamps.GenFiltMET<100 ) |  (allSamps.RunNumber!=410470)) & (allSamps.nsignalLep==0) & (allSamps.nbaselineLep==0) & (allSamps.eT_miss>250) & (allSamps.minDPhi_4jetMET>0.4) & (allSamps.num_cjets20>=2)

    rawDataPresel = allSamps[eval(presel)].copy()
    #rawDataPresel = allSamps[myPresel].copy()
    sampSizes = {}
    sampYields = {}

    for sampName in sorted(pd.unique(rawDataPresel.sampName)):
        sampIndex = rawDataPresel['sampName']==sampName
        sampYield = rawDataPresel[sampIndex].weight.sum()
        if sampYield == 0:
            print("Dropping", sampName)
            rawDataPresel.drop(sampIndex)
        sampSizes[sampName] = rawDataPresel[sampIndex].shape[0]
        sampYields[sampName] = sampYield
        print(sampName, round(sampYield,1), sampIndex.sum())
        
    maxYieldKey = max(sampYields, key=lambda k: sampYields[k])
    tempDFs = []
    goodSamps = pd.unique(rawDataPresel.sampName)
    for sampName in sampYields:
        sampIndex = rawDataPresel['sampName']==sampName
        rawDataPresel.loc[sampIndex, 'clus_weight'] = rawDataPresel.loc[sampIndex, 'weight']*(sampYields[maxYieldKey]/sampYields[sampName])
        #print(rawDataPresel[sampIndex].clus_weight.sum())

    # Tried adding more variables but this seems to cause a degradation in performance: only one signal cluster is found.
    # (two are expected based on the manually designed signal regions).
    trainBranches = ['MTcMin20', 'metsigST', 'm_cc20', 'pT_1jet', 'pT_2jet']#, 'pT_1cjet', 'pT_2cjet']# 'eT_miss']
    #trainBranches = ['MTcMin20', 'metsigST', 'm_cc20', 'pT_1cjet', 'pT_2cjet']#, 'eT_miss']
    varListStr = '_'.join(trainBranches)
    print(varListStr)

    scaledData = rawDataPresel.copy(deep=True)
    scalers = {}
    for column in trainBranches:
        print(column)
        scalers[column] = MinMaxScaler()
        scaledData[[column]] = scalers[column].fit_transform(scaledData[[column]])
    clusteringData = scaledData[trainBranches]

    twoSigs = scaledData[(scaledData.sampName=='sig_1300_1') | ( scaledData.sampName=='sig_550_375')]
    #bkgs = scaledData[(scaledData.sampName=='bkg')].sample(1000)
    # aeData = bkgs[trainBranches]
    # weights = bkgs['clus_weight']
    aeData = twoSigs[trainBranches]
    tempWeights = twoSigs['clus_weight']
    print(aeData.shape)

    batch_size = args.batch_size

    testTrainFrac = 0.5
    nEvents=aeData.shape[0]
    nTrain = np.floor(nEvents*testTrainFrac)
    newNTrain = int(np.floor(nTrain/batch_size)*batch_size)
    msk = np.random.choice(nEvents, newNTrain, replace=False)
    xTrain = aeData.to_numpy()[msk]

    xTest = aeData.to_numpy()[~msk]
    trainWeights = torch.Tensor(tempWeights.to_numpy()[msk])
    testWeights = torch.Tensor(tempWeights.to_numpy()[~msk])
    # trainWeights = torch.ones(msk.shape[0])
    # testWeights = torch.ones((~msk).shape[0])
    if torch.cuda.is_available():
        trainWeights = trainWeights.cuda()
        testWeights = testWeights.cuda()


    trainDataset = torch.utils.data.TensorDataset(torch.Tensor(xTrain),torch.Tensor(xTrain), trainWeights) # create your datset
    testDataset = torch.utils.data.TensorDataset(torch.Tensor(xTest), torch.Tensor(xTest), testWeights) # create your datset
    train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size) 

    # Main body
    model = DCN(args)
    rec_loss_list, mse_list = solver(
        args, model, train_loader, test_loader)
