import torch
from torchsummary import summary
import random
import os, sys
import argparse
import random
import numpy as np
from simple_ae import autoencoder
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
    
    for data, _, _ in test_loader:
        data = data.to(model.device)
        _, pred_data = model(data)
        mse.append(torch.nn.functional.mse_loss(torch.as_tensor(pred_data.detach().cpu().numpy()), torch.as_tensor(data.detach().cpu().numpy())))
    
    mse = np.mean(mse)

    return mse

def solver(args, model, train_loader, test_loader, signals):

    train_mse_list = []
    test_mse_list = []

    for e in range(args.epoch):
        model.train()

        # throughput measurement - start
        if args.time_throughput:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

        model.fit(e, train_loader)

        # throughput measurement - end
        if args.time_throughput:
            ender.record()
            torch.cuda.synchronize()
            globals.total_throughput_time += starter.elapsed_time(ender)/1000
        
        model.eval()
        # evaluate on train data
        MSE = evaluate(model, train_loader)
        train_mse_list.append(MSE)
        # evaluate on test data
        MSE = evaluate(model, test_loader)
        test_mse_list.append(MSE)
        
        print('\nEval Epoch: {:02d} | AE MSE: {:.5f} \n'.format(e, MSE))

    return train_mse_list, test_mse_list

#################################################

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

    # Model parameters
    parser.add_argument('--latent-dim', type=int, default=2,
                        help='latent space dimension')

    # Utility parameters
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
    print(f'latent_dim:         {args.latent_dim}')
    print(f'cuda:               {args.cuda}')
    print(f'log_interval:       {args.log_interval}')
    print(f'time_latancy:       {args.time_latency}')
    print(f'time_throughput:    {args.time_throughput}')
    print(f'randseed:           {args.randseed}')

    # fix random seed
    run_deterministic = True
    if args.time_latency or args.time_throughput: 
        run_deterministic = False
    if args.randseed != 0:
        fix_randomness(args.randseed, True)

    # set signals and variables to operate on
    # signals = ['sig_1300_1', 'sig_550_375']
    signals = ['sig_1000_1', 'sig_1000_100', 'sig_1000_200', 'sig_1000_300', 'sig_1000_400', 'sig_1000_500', 'sig_1000_600', 'sig_1000_700', 'sig_1050_150', 'sig_1050_250', 'sig_1050_50', 'sig_1100_1', 'sig_1100_100', 'sig_1100_200', 'sig_1100_300', 'sig_1100_400', 'sig_1100_500', 'sig_1100_600', 'sig_1100_700', 'sig_1200_1', 'sig_1200_100', 'sig_1200_200', 'sig_1200_300', 'sig_1200_400', 'sig_1200_500', 'sig_1200_600', 'sig_1300_1', 'sig_1300_100', 'sig_1300_200', 'sig_1300_300', 'sig_1300_400', 'sig_1300_500', 'sig_400_225', 'sig_500_1', 'sig_500_325', 'sig_550_375', 'sig_600_1', 'sig_600_300', 'sig_600_425', 'sig_650_350', 'sig_650_475', 'sig_700_1', 'sig_700_100', 'sig_700_200', 'sig_700_300', 'sig_700_400', 'sig_700_525', 'sig_750_250', 'sig_750_350', 'sig_750_450', 'sig_750_575', 'sig_800_1', 'sig_800_100', 'sig_800_200', 'sig_800_300', 'sig_800_400', 'sig_800_500', 'sig_800_625', 'sig_850_150', 'sig_850_250', 'sig_850_350', 'sig_850_450', 'sig_850_50', 'sig_900_1', 'sig_900_100', 'sig_900_200', 'sig_900_300', 'sig_900_400', 'sig_900_500', 'sig_900_600', 'sig_950_150', 'sig_950_250', 'sig_950_350', 'sig_950_50']
    trainBranches = ['MTcMin20', 'metsigST', 'eT_miss', 'nj_good', 'pT_1jet', 'pT_2jet', 'num_bjets', 'num_cjets20', 'sampName'] # sampName should always stay here
    # re-set the input dimentions
    args.input_dim = len(trainBranches)-1
    # get the train/test DataLoaders
    train_loader, test_loader = get_DataLoaders(args, signals, trainBranches)

    print('number of train batches:', len(train_loader))
    print('number of test batches:', len(test_loader))

    #################################################
    # Main body

    #  use gpu if available
    globals.device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")

    # create a model and load it to device
    model = autoencoder(args).to(globals.device)

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
        _, _ = model(inputs) # warm-up
        torch.cuda.synchronize() # wait for warm-up to finish

        # get random training tensor of batch = 1
        input = inputs[random.randint(0, args.batch_size-1), :].view(1,-1)
        # time it
        with torch.no_grad():
            model.eval()
            for rep in range(repetitions):
                starter.record() # start stopwatch
                _, _ = model(input)
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
    train_mse_list, test_mse_list = solver(args,
                                           model,
                                           train_loader,
                                           test_loader,
                                           signals)

    if args.time_throughput:
        print("Using batch-size: %i instances" % (globals.timing_batch_size))
        print("Measured throughput: %i instances/sec" % (globals.timing_epochs*globals.timing_batch_size*len(train_loader)/globals.total_throughput_time))
        print("Measured throughput (no data-transfer): %i instances/sec" % (globals.timing_epochs*globals.timing_batch_size*len(train_loader)/globals.total_throughput_time_noDT))
