'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import shutil
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim.lr_scheduler import _LRScheduler
from torch.autograd import Variable
from collections import OrderedDict
from sklearn.neighbors import NearestNeighbors

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


#_, term_width = os.popen('stty size', 'r').read().split()
#term_width = int(term_width)
term_width, rows = shutil.get_terminal_size()

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]



def density_entropy(X):
    K = 5
    N, C, D = X.shape
    x = X.transpose(1, 0, 2).reshape(C, N, -1)
    score = []
    for c in range(C):
        nbrs = NearestNeighbors(n_neighbors=K + 1).fit(x[c])
        dms = []
        for i in range(N):
            dm = 0
            dist, ind = nbrs.kneighbors(x[c, i].reshape(1, -1))
            for j, id in enumerate(ind[0][1:]):
                dm += dist[0][j + 1]

            dms.append(dm)

        dms_sum = sum(dms)
        en = 0
        for i in range(N):
            en += -dms[i]/dms_sum*math.log(dms[i]/dms_sum, 2)

        score.append(en)
    return np.array(score)


def info_richness(model, conv_list):
    model.eval()
    for i in conv_list:
        weight = model.features[i].weight.data.cpu()
        in_channels, out_channels,_,_ = weight.shape
        weight = weight.numpy().transpose(1, 0, 2, 3)
        entropy = density_entropy(weight.reshape(out_channels, in_channels, -1))
        print(model.features[i])
        print(max(entropy),min(entropy))

# Function to return Number of parameters & Multiplications
def summary(model, input_size):
    #inspired from https://github.com/sksq96/pytorch-summary/blob/master/torchsummary/torchsummary.py
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)
                if module_idx == 261:
                    print(str(module.__class__))

                m_key = '%s-%i' % (class_name, module_idx+1)
                summary[m_key] = OrderedDict()
                summary[m_key]['input_shape'] = list(input[0].size())
                summary[m_key]['input_shape'][0] = -1
                if isinstance(output, (list,tuple)):
                    summary[m_key]['output_shape'] = [[-1] + list(o.size())[1:] for o in output]
                else:
                    summary[m_key]['output_shape'] = list(output.size())
                    summary[m_key]['output_shape'][0] = -1

                wt_params = 0
                if hasattr(module, 'weight') and 'conv' in m_key.lower():
                    wt_params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]['trainable'] = module.weight.requires_grad
                bias_params = 0
                if hasattr(module, 'bias') and hasattr(module.bias, 'size') and 'conv' in m_key.lower():
                    bias_params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]['nb_params'] = wt_params + bias_params
                
                if 'conv' in m_key.lower():
                    summary[m_key]['MulC'] = wt_params * summary[m_key]['output_shape'][-1] * summary[m_key]['output_shape'][-2]
                elif 'linear' in m_key.lower():
                    summary[m_key]['MulC'] = wt_params
                else:
                    summary[m_key]['MulC'] = 0

            if (not isinstance(module, nn.Sequential) and 
               not isinstance(module, nn.ModuleList) and 
               not (module == model)):
                hooks.append(module.register_forward_hook(hook))
                
        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        
        # check if there are multiple inputs to the network
        if isinstance(input_size[0], (list, tuple)):
            x = [Variable(torch.rand(1,*in_size)).type(dtype) for in_size in input_size]
        else:
            x = Variable(torch.rand(1,*input_size)).type(dtype)
            
            
        # print(type(x[0]))
        # create properties
        summary = OrderedDict()
        hooks = []
        # register hook
        model.apply(register_hook)
        # make a forward pass
        # print(x.shape)
        model.cuda()
        print(next(model.parameters()).is_cuda)
        model(x)
        # remove these hooks
        for h in hooks:
            h.remove()

        #print('----------------------------------------------------------------')
        line_new = '{:>20}  {:>25} {:>15} {:>20}'.format('Layer (type)', 'Output Shape', 'Param #', 'Muls #')
        print(line_new)
        #print('======================================================================================')
        total_params = 0
        trainable_params = 0
        total_comp = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = '{:>20}  {:>25} {:>15} {:>20}'.format(layer, str(summary[layer]['output_shape']), summary[layer]['nb_params'], summary[layer]['MulC'])
            total_params += summary[layer]['nb_params']
            total_comp   += summary[layer]['MulC']
            if 'trainable' in summary[layer]:
                if summary[layer]['trainable'] == True:
                    trainable_params += summary[layer]['nb_params']
            print(line_new)
        print('======================================================================================')
        print('Total params: ' + str(total_params))
        print('Trainable params: ' + str(trainable_params))
        print('Non-trainable params: ' + str(total_params - trainable_params))
        print('Total Multiplications: ' + str(total_comp))
        print('----------------------------------------------------------------')        
        line_new = '{:>25}  {:>25}'.format(str(trainable_params.numpy()), str(total_comp.numpy()))
        print(line_new)
        return trainable_params, total_comp
        return trainable_params, total_comp
