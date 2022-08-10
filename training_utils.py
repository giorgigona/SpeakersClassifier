from torch.utils.data import Dataset
import torch


class SpeechDataLoader(Dataset):
    """
    Pytorch dataloader class

    Args:
        data (torch.Tensor): spectrograms tensor
        labels (torch.Tensor)): speakers labels array
    """

    def __init__(self, data, labels):
        super().__init__()
        self.data = data
        self.labels = labels
            
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, idx):
        waveform = self.data[idx]
        out_labels = self.labels[idx]
        return waveform, out_labels

def train_model(
    model, 
    trainloader, 
    validloader,
    optimizer, 
    criterion, 
    epoch, 
    device, 
    verbose=True):

    """
    Train a pytorch nn.Module

    Args:
        model: model to be trained
        trainloader: training data dataloader
        validloader: validation data dataloader
        optimizer: optimizer for model
        criterion: training criterion for model
        epoch (int): number of epoch
        device (str): device to train model
        verbose (bool, optional=True): bool to print training info
    """

    model.train()
    train_loss = 0
    total = 0
    total_correct = 0
    
    iterator = iter(trainloader)
    
    for inputs, targets in iterator:
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _,predicted = torch.max(outputs.data,1)
        total_correct += (predicted == targets).sum().item()
        total += targets.size(0)

    # print results
    if verbose:
        if validloader is None:
            print("Epoch: [{}]  Training Loss: [{:.3f}] Train Accuracy [{:.3f}] ".format(
                epoch + 1, 
                train_loss/len(trainloader),
                total_correct*100/total)
            )
        else:
            valid_accuracy = test_model(model, criterion, validloader, device)
            print("Epoch: [{}]  Training Loss: [{:.3f}] Train Accuracy [{:.3f}] Valid Loss [{:.3f}] Valid Accuracy [{:.3f}]".format(
                epoch + 1, 
                train_loss/len(trainloader),
                total_correct*100/total,
                valid_accuracy['loss'],
                valid_accuracy['accuracy'])
            )

    

def test_model(model, criterion, testloader, device):
    """
    Test model performance

    Args:
        model (nn.Module): trained model
        criterion (function): evaluation criterion
        testloader (DataLoader): test data dataloader
        device (str): device for model

    Returns:
        dict: dictionary of evaluation results
    """
    test_loss, total, total_correct = 0,0,0

    iterator = iter(testloader)
    
    for inputs, targets in iterator:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        total_correct += (predicted == targets).sum().item()

    accuracy = 100. * total_correct / total

    return {'loss': loss, 'accuracy': accuracy}        
    