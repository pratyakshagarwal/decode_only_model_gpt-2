import torch
import torch.nn as nn
import torch.nn.functional as F

#######################################################################################################################
class FeedForward(nn.Module):
    """
    FeedForward module in the Transformer encoder layer.
    
    Args:
        n_embed (int): Size of the input and output embeddings.
        dropout (float): Dropout probability.
        expand (int): Expansion factor for the hidden layer size. Default is 4.
    """
    def __init__(self, n_embed, dropout, expand=4):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, expand * n_embed),
            nn.ReLU(),
            nn.Linear(expand * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass of the FeedForward module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C), where
                              B is the batch size, T is the sequence length,
                              and C is the number of input channels.

        Returns:
            torch.Tensor: Output tensor after the feedforward operation, of shape (B, T, C).
        """
        return self.net(x)
    
#############################################################################################################################

if __name__ == '__main__':
    pass