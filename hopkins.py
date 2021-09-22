import torch
import random
import os
import argparse
import random
import os
import numpy as np
import pandas as pd
from DCN import DCN
from sklearn.preprocessing import MinMaxScaler
# from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pdb
import fnmatch,collections

selections=collections.OrderedDict()

# set random seeds
def fix_randomness(seed: int, deterministic: bool = False) -> None:
    # pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

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

def evaluate(model, test_loader, signals=None):
    mse = []
    targets_true = []
    cluster_pred = []
    latent_data = []
    for data, target, weights in test_loader:
        data = data.to(model.device)
        latent_X = model.autoencoder(data, latent=True)
        latent_X = latent_X.detach().cpu().numpy()
        pred_data = model.autoencoder(data)
        pred_data = pred_data.detach().cpu().numpy()

        mse.append(torch.nn.functional.mse_loss(torch.as_tensor(pred_data),torch.as_tensor(data.detach().cpu().numpy())))
        targets_true.append(target)
        cluster_pred.append(model.kmeans.update_assign(latent_X).reshape(-1, 1))
        latent_data.append(latent_X)

    targets_true = np.vstack(targets_true).astype(int).reshape(-1)
    cluster_pred = np.vstack(cluster_pred).astype(int).reshape(-1)
    latent_data  = np.vstack(latent_data)

    cluster_diff = (targets_true - cluster_pred).mean()

    if signals is not None:
        unq = np.unique(targets_true)
        fig,ax = plt.subplots(len(unq),1,figsize=(12,8*len(unq)),dpi=80)

        x = latent_data[:,0]
        y = latent_data[:,1]
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        for i in unq:
            ax[i].hist2d(x[targets_true == i],y[targets_true == i],100,[[x_min,x_max],[y_min,y_max]],cmap='Reds')
            ax[i].set_title(signals[i])
            ax[i].set_xlabel('latent[0]')
            ax[i].set_ylabel('latent[1]')

        fig.savefig('evaluate.png')
        plt.close('all')

    mse = np.mean(mse)
    return mse, cluster_diff

def solver(args, model, train_loader, test_loader, signals):

    rec_loss_list = model.pretrain(train_loader, args.pre_epoch)
    print('pretrain clusters: ',model.kmeans.clusters)

    train_mse_list = []
    train_cdiff_list = []
    test_mse_list = []
    test_cdiff_list = []

    for e in range(args.epoch):
        model.train() 
        model.fit(e, train_loader)
        
        model.eval()
        # evaluate on train data
        MSE, CDIFF = evaluate(model, train_loader)
        train_mse_list.append(MSE)
        train_cdiff_list.append(CDIFF)
        # evaluate on test data
        MSE, CDIFF = evaluate(model, test_loader, signals)
        test_mse_list.append(MSE)
        test_cdiff_list.append(CDIFF)
        
        print('\nEval Epoch: {:02d} | AE MSE: {:.5f} | CLUSTER ACCURACY: {:.5f} \n'.format(e, MSE,CDIFF))

    return rec_loss_list, train_mse_list, train_cdiff_list, test_mse_list, test_cdiff_list

def plot_and_save(name, test_data_list, train_data_list=None):
    plt.clf()
    plt.plot(test_data_list, linewidth=2, label='test')
    if train_data_list is not None:
        plt.plot(train_data_list, linewidth=2, label='train')
    plt.legend()
    plt.xlabel("Epoch")
    plt.yscale('log')
    
    if name == 'MSE':
        ylabel = "MSE Reconstruction Loss"
        savename = 'reco_loss.png'

    elif name == 'CDIFF':
        ylabel = "Mean Clusters Difference"
        savename = 'cdiff.png'
    
    plt.ylabel(ylabel)
    plt.savefig(savename, bbox_inches='tight')

# set random seeds
def fix_randomness(seed: int, deterministic: bool = False) -> None:
    # pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Deep Clustering Network')

    # Dataset parameters
    parser.add_argument('--data', default='/home/ekourlitis/group/ccMET_noBackground.h5',
                        help='dataset path')
    parser.add_argument('--input-dim', type=int, default=5,
                        help='input dimension')

    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--batch-size', type=int, default=25,
                        help='input batch size for training')
    parser.add_argument('--epoch', type=int, default=30,
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
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[500, 500, 2000],
                        help='encoder-decoder Linear layer shapes')
    parser.add_argument('--latent-dim', type=int, default=2,
                        help='latent space dimension')
    parser.add_argument('--n-clusters', type=int, default=2,
                        help='number of clusters in the latent space')
    parser.add_argument('--activation', default='LeakyReLU',
                        help='activation for autoencoder')

    # Utility parameters
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='number of jobs to run in parallel')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='whether to use GPU')
    parser.add_argument('--log-interval', type=int, default=100,
                        help=('how many batches to wait before logging the '
                              'trainin````g status'))
    parser.add_argument('--timing', action='store_true',
                        help=('timing measurements configuration'))
    parser.add_argument('--randseed',type=int,default=0,help='if set to nonzero value, fixes random seed in torch and numpy')

    args = parser.parse_args()

    # print parameters
    print(f'data: {args.data}')
    print(f'input_dim:      {args.input_dim}')
    print(f'lr:             {args.lr}')
    print(f'wd:             {args.wd}')
    print(f'batch_size:     {args.batch_size}')
    print(f'epoch:          {args.epoch}')
    print(f'pre_epoch:      {args.pre_epoch}')
    print(f'pretrain:       {args.pretrain}')
    print(f'lamda:          {args.lamda}')
    print(f'beta:           {args.beta}')
    print(f'hidden_dims:    {args.hidden_dims}')
    print(f'latent_dim:     {args.latent_dim}')
    print(f'n_clusters:     {args.n_clusters}')
    print(f'activation:     {args.activation}')
    print(f'n_jobs:         {args.n_jobs}')
    print(f'cuda:           {args.cuda}')
    print(f'log_interval:   {args.log_interval}')
    print(f'randseed:       {args.randseed}')

    # fix random seed
    run_deterministic = True
    if args.timing: 
        run_deterministic = False
    if args.randseed != 0:
        fix_randomness(args.randseed, True)

    # Load data
    allSamps = pd.read_hdf(args.data)

    # get selection
    presel = selections['presel_0lep_1cjet20'].replace('*', ' & ').replace('||', ' | ').replace('(', '(allSamps.').replace('allSamps.(', '(')
    print(presel)
    #print(sorted(branches))
    myPresel = (allSamps.passTightCleanDFFlag==1) & (allSamps.nj_good>=2) & ((allSamps.tcMeTCategory==1) | (allSamps.tcMeTCategory<0)) & (allSamps.pT_1jet>250) & (allSamps.num_bjets==0) & ((allSamps.GenFiltMET<100 ) |  (allSamps.RunNumber!=410470)) & (allSamps.nsignalLep==0) & (allSamps.nbaselineLep==0) & (allSamps.eT_miss>250) & (allSamps.minDPhi_4jetMET>0.4) & (allSamps.num_cjets20>=2)

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

    # define training variables
    # Tried adding more variables but this seems to cause a degradation in performance: only one signal cluster is found.
    # (two are expected based on the manually designed signal regions).
    trainBranches = ['MTcMin20', 'metsigST', 'm_cc20', 'pT_1jet', 'pT_2jet', 'sampName']#, 'pT_1cjet', 'pT_2cjet']# 'eT_miss']
    varListStr = '_'.join(trainBranches)
    # print('varListStr:',varListStr)

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

    # data for clustering
    clusteringData = scaledData[trainBranches]

    # select signal samples
    signals = ['sig_1300_1','sig_550_375'] #, 'sig_900_600']
    print('using only samples: ',signals)
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

    # batch size to be used
    batch_size = args.batch_size

    # test/train fraction
    testTrainFrac = 0.7

    # select events
    nEvents=aeData.shape[0]
    nTrain = np.floor(nEvents*testTrainFrac)
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
    trainDataset = torch.utils.data.TensorDataset(torch.Tensor(train_inputs),torch.Tensor(train_targets), trainWeights) # create your datset
    testDataset = torch.utils.data.TensorDataset(torch.Tensor(test_inputs), torch.Tensor(test_targets), testWeights) # create your datset
    
    # torch DataLoaders
    train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size) 

    print('number of train batches:',len(train_loader))
    print('number of test batches:',len(test_loader))

    # Main body
    model = DCN(args)

    #################################################
    
    if args.timing:
        # init loggers
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 1000
        timings = np.zeros(repetitions)

        # pass random training tensor to wamp-up GPU
        dataiter = iter(train_loader)
        inputs, _, _ = dataiter.next()
        inputs = inputs.cuda().view(inputs.shape[0], -1)
        _ = model.autoencoder(inputs) # warm-up
        torch.cuda.synchronize() # wait for warm-up to finish

        # get random training tensor of batch = 1
        input = inputs[random.randint(0, batch_size-1), :].view(1,-1)
        # time it
        with torch.no_grad():
            model.eval()
            for rep in range(repetitions):
                starter.record() # start stopwatch
                _ = model.autoencoder(input)
                ender.record() # stop stopwatch
                # wait for GPU to sync
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

        print("Mean latency: %0.4f ms" % (np.mean(timings)))
        print("Std latency: %0.4f ms" % (np.std(timings)))

        exit(0)

    #################################################

    # train
    rec_loss_list, train_mse_list, train_cdiff_list, test_mse_list, test_cdiff_list = solver(args,
                                                                                             model,
                                                                                             train_loader,
                                                                                             test_loader,
                                                                                             signals)

    # plot metrics calculated at evaluation on test_loader
    plot_and_save('MSE', test_mse_list, train_mse_list)
    plot_and_save('CDIFF', test_cdiff_list)
