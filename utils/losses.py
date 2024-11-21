import torch
import torch.nn.functional as F

def mse_loss(input, target):
    """
    Mean Squared Error Loss
    
    Parameters
    ----------
    input : torch.Tensor
        The input tensor.
    target : torch.Tensor
        The target tensor.
        
    Returns
    -------
    torch.Tensor
        The mean squared error loss between the input and target tensors.
    """
    return F.mse_loss(input, target)

def mae_loss(input, target):
    """
    Mean Absolute Error Loss
    
    Parameters
    ----------
    input : torch.Tensor
        The input tensor.
    target : torch.Tensor
        The target tensor.
        
    Returns
    -------
    torch.Tensor
        The mean absolute error loss between the input and target tensors.# Below is the code of /home/kimariyb/project/SABRE-Denoise/utils/losses.py 
    """
    return F.l1_loss(input, target)

def nmse_loss(input, target):    
    """
    Normalized Mean Squared Error Loss
    
    Parameters
    ----------
    input : torch.Tensor
        The input tensor.
    target : torch.Tensor
        The target tensor.
            
    Returns
    -------
    torch.Tensor
        The normalized mean squared error loss between the input and target tensors. 
    """
    return torch.mean((input - target) ** 2) / torch.mean(target ** 2)  
    
def rmse_loss(input, target):
    """
    Root Mean Squared Error Loss
    
    Parameters
    ----------
    input : torch.Tensor
        The input tensor.
    target : torch.Tensor
        The target tensor.
        
    Returns
    -------
    torch.Tensor
        The root mean squared error loss between the input and target tensors.  # Below is the code of /home/kimariyb/project/SABRE-Denoise/utils/losses.py  # Below is the code of /home/kimariyb
    """
    return torch.sqrt(F.mse_loss(input, target))

def huber_loss(input, target, delta=1.0):
    """
    Huber Loss
    
    Parameters
    ----------    
    input : torch.Tensor
        The input tensor.
    target : torch.Tensor
        The target tensor.
    delta : float
        The threshold at which to change between L1 and L2 loss.
        
    Returns
    -------
    torch.Tensor
        The Huber loss between the input and target tensors.
    """
    return F.smooth_l1_loss(input, target, beta=delta)

def log_cosh_loss(input, target):
    """
    Log-Cosh Loss
    
    Parameters
    ----------    
    input : torch.Tensor
        The input tensor.
    target : torch.Tensor
        The target tensor.
        
    Returns
    -------
    torch.Tensor
        The log-cosh loss between the input and target tensors.
    """
    return torch.mean(torch.log(torch.cosh(input - target)))