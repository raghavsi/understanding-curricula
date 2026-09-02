import argparse
from itertools import combinations
from tqdm import tqdm
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn.functional as F

from torchvision import models, datasets, transforms
from typing import Any, Callable, Optional, Tuple, Union
import numpy as np
from PIL import Image

def variance_metrics(all_grads):
    cosine_dist = cosine_distance(all_grads)
    #variance, spectral_norm = raw_variance_and_spectral_norm(all_grads)
    return cosine_dist, 0,0#variance, spectral_norm

def cosine_distance(grads):
    running_dist = 0
    for i, j in combinations(range(len(grads)), 2):
        cosine_similiarity = np.dot(grads[i], grads[j])/(np.linalg.norm(grads[i])*np.linalg.norm(grads[j]))
        running_dist += 1 - cosine_similiarity
    print(running_dist)
    return running_dist / (len(grads)*(len(grads)-1))

def raw_variance_and_spectral_norm(grads):
    cov = calculate_covariance(grads)
    trace =np.trace(cov)
    spectral_norm = np.linalg.norm(cov, ord = 2)
    return trace, spectral_norm

def calculate_covariance(X):
    mean = np.mean(X, axis=0, keepdims=True)
    X = X - mean
    return 1/X.shape[0] * X @ X.transpose(-1, -2)

def get_CIFAR10(root="./"):
    input_size = 32
    num_classes = 10
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = datasets.CIFAR10(
        root + "CIFAR10", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_dataset = datasets.CIFAR10(
        root + "CIFAR10", train=False, transform=test_transform, download=True
    )

    return input_size, num_classes, train_dataset, test_dataset

import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18(num_classes=10)

        self.resnet.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.maxpool = torch.nn.Identity()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        x = F.log_softmax(x, dim=1)

        return x
    
class ModelPT(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')

        # self.resnet.conv1 = torch.nn.Conv2d(
        #     3, 64, kernel_size=3, stride=1, padding=1, bias=False
        # )
        # self.resnet.maxpool = torch.nn.Identity()
        num_ftrs = self.resnet.fc.in_features

        self.resnet.fc = nn.Linear(num_ftrs,10)

    def forward(self, x):
        x = self.resnet(x)
        x = F.log_softmax(x, dim=1)

        return x
    
def train(model, train_loader, optimizer, epoch):
    model.train()

    total_loss = []
    all_grads = []
    for data, target in tqdm(train_loader):
        data = data.to("cuda")
        target = target.to("cuda")

        optimizer.zero_grad()

        prediction = model(data)
        #prediction = F.log_softmax(prediction, dim=1)

        loss = F.nll_loss(prediction, target)

        loss.backward()
        # grads = []
        # for weight in model.parameters():
        #   grads.append(weight.grad.reshape(-1))
        
        # all_grads.append(torch.cat(grads).detach().cpu().numpy())
        optimizer.step()

        total_loss.append(loss.item())

    avg_loss = sum(total_loss) / len(total_loss)
    print(f"Epoch: {epoch}:")
    print(f"Train Set: Average Loss: {avg_loss:.8f}")
    return all_grads
    
def test(model, test_loader):
    model.eval()

    loss = 0
    correct = 0

    for data, target in test_loader:
        with torch.no_grad():
            data = data.to("cuda")
            target = target.to("cuda")

            prediction = model(data)
            #prediction = F.log_softmax(prediction, dim=1)
            loss += F.nll_loss(prediction, target, reduction="sum")

            prediction = prediction.max(1)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    loss /= len(test_loader.dataset)

    percentage_correct = 100.0 * correct / len(test_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            loss, correct, len(test_loader.dataset), percentage_correct
        )
    )

    return loss, percentage_correct

from functorch.experimental import replace_all_batch_norm_modules_
from functorch import make_functional_with_buffers, vmap, grad
def get_grad_norms(params, buffers, fixed_loader):
    fmodel.eval()

    grad_norms = []

    for data_batch, target_batch in tqdm(fixed_loader):
        data_batch = data_batch.to("cuda")
        target_batch = target_batch.to("cuda")

        ft_per_sample_grads = ft_compute_sample_grad(params, buffers, data_batch, target_batch)

        squared_norm = 0
        for param_grad in ft_per_sample_grads:
            #print('a',param_grad.shape)
            squared_norm += param_grad.flatten(1).square().sum(dim=-1)
        grad_norms.append(squared_norm.detach().cpu().numpy()**0.5)

    grad_norms = np.concatenate(grad_norms, axis=0)
    return grad_norms

def compute_loss_stateless_model (params, buffers, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)

    predictions = fmodel(params, buffers, batch)
    loss = F.nll_loss(predictions, targets)
    return loss
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def iterate_dataset(dataset: Dataset, batch_size: int):
    """Iterate through a dataset, yielding batches of data."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for (batch_X, batch_y) in loader:
        yield batch_X.cuda(), batch_y.cuda()
def compute_gradient(network: nn.Module, loss_fn: nn.Module,
                     dataset: Dataset, physical_batch_size: int = 128):
    """ Compute the gradient of the loss function at the current network parameters. """
    p = len(parameters_to_vector(network.parameters()))
    average_gradient = torch.zeros(p, device='cuda')
    batch_grad_list = []
    count = 0
    norm_gradient = torch.zeros(p, device='cuda')
    for (X, y) in iterate_dataset(dataset, physical_batch_size):
        batch_loss = loss_fn(network(X), y) / len(dataset)
        batch_gradient = parameters_to_vector(torch.autograd.grad(batch_loss, inputs=network.parameters()))
        count+=1
        average_gradient += batch_gradient
        #print('b',batch_gradient.shape)
        norm_gradient += batch_gradient.square()
        if np.random.rand()>0.85:
            batch_grad_list.append(batch_gradient.detach().cpu().numpy())
    return batch_grad_list, average_gradient/count, norm_gradient.sum()/count



if __name__ == '__main__':
    torch.manual_seed(2048)

    # get CIFAR10 dataset
    input_size, num_classes, train_dataset, test_dataset = get_CIFAR10()
    kwargs = {"num_workers": 2, "pin_memory": True}

    fixed_train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=False, **kwargs
    )
    train_loader = torch.utils.data.DataLoader(
       train_dataset, batch_size=128, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False, **kwargs
    )

    # create model
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=5e-4,nesterov=True)
    #optimizer = optim.RMSprop(net.parameters(),lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    fmodel, params, buffers = make_functional_with_buffers(model)
    ft_compute_grad = grad(compute_loss_stateless_model)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

    import scipy
    loss_fn = nn.NLLLoss(reduction = 'sum')
    G = []
    for epoch in range(20):
        if epoch%5 ==0:
            loss = test(model.to("cuda"),test_loader)
            print('Test loss',epoch, loss,)
        if epoch%5==0:
            _, params, buffers = make_functional_with_buffers(model.to("cuda"))
            grad_norms = get_grad_norms(params, buffers, fixed_train_loader)
            print("mean gradient norn",np.mean(grad_norms))
            b,avg_grad,norm_grad =  compute_gradient(model.cuda(),loss_fn ,train_dataset)
            print("avg",norm_grad)
            G.append(b) 
            
        all_grads = train(model.to("cuda"), train_loader, optimizer, epoch)
        to_save = {
            'model': model.state_dict()
        }
        torch.save(to_save,'model_'+str(epoch)+'.pt')
        # all_grads = np.stack(all_grads, axis = 0)
        # print(np.mean(all_grads))
        # cosine_dist, variance, spectral_norm = variance_metrics(all_grads)
        # print(f'* Variance Stats: Cosine Distance {cosine_dist:.3f} Variance {variance:.3f} Spectral Norm {spectral_norm:.3f}')
       
        #filter = lambda p: not None or len(p.data.size()) > 1
        #x = np.concatenate([p.grad.data.cpu().numpy().ravel() for p in model.parameters() if filter(p)])
        #print(x.shape)

        scheduler.step()
    update_freq = 1
    optimizer.zero_grad()
    for epoch in range(20):
        checkpoint = torch.load('model_'+str(epoch)+'.pt', map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        all_grads = []
        for inner_data_iter_step, (data, target) in enumerate(train_loader):
            #inner_step = inner_data_iter_step // update_freq
            #optimizer.curr_step = inner_step
            data = data.to("cuda")
            target = target.to("cuda")
            output = model(data)
            loss = F.nll_loss(output, target)
            loss /= update_freq
            loss.backward()
            grads = []
            for weight in model.parameters():
                grads.append(weight.grad.reshape(-1))
            all_grads.append(torch.cat(grads).detach().cpu().numpy())
            optimizer.zero_grad()
        all_grads = np.stack(all_grads, axis = 0)#torch.stack(all_grads, dim = 0)
        cosine_dist, variance, spectral_norm = variance_metrics(all_grads)
        print(f'* Variance Stats: Cosine Distance {cosine_dist:.3f} Variance {variance:.3f} Spectral Norm {spectral_norm:.3f}')
