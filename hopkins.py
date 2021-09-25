import torch
import random
import os, sys
import argparse
import random
import numpy as np
from DCN import DCN
# from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pdb
from data import get_DataLoaders
import globals
globals.init()

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
    # wait for pretrain to finish
    if args.time_throughput:
            torch.cuda.synchronize()
    print('pretrain clusters: ',model.kmeans.clusters)

    train_mse_list = []
    train_cdiff_list = []
    test_mse_list = []
    test_cdiff_list = []

    for e in range(args.epoch):
        model.train()

        # throughput measurement - start
        if args.time_throughput:
            starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
            starter.record()

        model.fit(e, train_loader)

        # throughput measurement - end
        if args.time_throughput:
            ender.record()
            torch.cuda.synchronize()
            globals.total_throughput_time += starter.elapsed_time(ender)/1000
        
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
    parser.add_argument('--time-latency', action='store_true',
                        help=('latency timing measurement configuration'))
    parser.add_argument('--time-throughput', action='store_true',
                        help=('throughput timing measurement configuration'))
    parser.add_argument('--randseed',type=int,default=0,help='if set to nonzero value, fixes random seed in torch and numpy')

    args = parser.parse_args()

    # print parameters
    print(f'data:               {args.data}')
    print(f'input_dim:          {args.input_dim}')
    print(f'lr:                 {args.lr}')
    print(f'wd:                 {args.wd}')
    print(f'batch_size:         {args.batch_size}')
    print(f'epoch:              {args.epoch}')
    print(f'pre_epoch:          {args.pre_epoch}')
    print(f'pretrain:           {args.pretrain}')
    print(f'lamda:              {args.lamda}')
    print(f'beta:               {args.beta}')
    print(f'hidden_dims:        {args.hidden_dims}')
    print(f'latent_dim:         {args.latent_dim}')
    print(f'n_clusters:         {args.n_clusters}')
    print(f'activation:         {args.activation}')
    print(f'n_jobs:             {args.n_jobs}')
    print(f'cuda:               {args.cuda}')
    print(f'log_interval:       {args.log_interval}')
    print(f'randseed:           {args.randseed}')
    print(f'time_latancy:       {args.time_latency}')
    print(f'time_throughput:    {args.time_throughput}')

    # fix random seed
    run_deterministic = True
    if args.time_latency or args.time_throughput: 
        run_deterministic = False
    if args.randseed != 0:
        fix_randomness(args.randseed, True)

    # set signals and variables to operate on
    signals = ['sig_1300_1', 'sig_550_375']
    trainBranches = ['MTcMin20', 'metsigST', 'eT_miss', 'nj_good', 'pT_1jet', 'pT_2jet', 'num_bjets', 'num_cjets20', 'sampName'] # sampName should always stay here
    # re-set the input dimentions
    args.input_dim = len(trainBranches)-1
    # get the train/test DataLoaders
    train_loader, test_loader = get_DataLoaders(args, signals, trainBranches)

    print('number of train batches:',len(train_loader))
    print('number of test batches:',len(test_loader))

    # Main body
    model = DCN(args)

    #################################################
    
    if args.time_latency:
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

        sys.exit()

    #################################################

    # train
    rec_loss_list, train_mse_list, train_cdiff_list, test_mse_list, test_cdiff_list = solver(args,
                                                                                             model,
                                                                                             train_loader,
                                                                                             test_loader,
                                                                                             signals)

    if args.time_throughput:
        print("Measured throughput: %i instances/sec" % (args.epoch*batch_size/globals.total_throughput_time))

    # plot metrics calculated at evaluation on test_loader
    if not args.time_throughput:
        plot_and_save('MSE', test_mse_list, train_mse_list)
        plot_and_save('CDIFF', test_cdiff_list)
