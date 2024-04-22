import torch
import torch.nn as nn
import torch.nn.functional as F
from .multiheadattn import MultiAttentionHead
from .feedforward import FeedForward

########################################################################################################################
class Block(nn.Module):
    """
    Transformer block containing a Multi-Head Attention layer followed by a FeedForward layer.
    
    Args:
        n_heads (int): Number of attention heads.
        n_embed (int): Size of the input and output embeddings.
        block_size (int): Block size for the attention mechanism.
        dropout (float): Dropout probability.
        expand (int): Expansion factor for the hidden layer size.
    """
    def __init__(self, n_heads, n_embed, block_size, dropout, expand):
        super(Block, self).__init__()
        self.sa = MultiAttentionHead(n_heads, n_embed, block_size, dropout)
        self.ffn = FeedForward(n_embed, dropout, expand)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        """
        Forward pass of the Transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C), where
                              B is the batch size, T is the sequence length,
                              and C is the number of input channels.

        Returns:
            torch.Tensor: Output tensor after the Transformer block, of shape (B, T, C).
        """
        # Multi-Head Attention layer
        x = x + self.sa(self.ln1(x))
        # FeedForward layer
        x = x + self.ffn(self.ln2(x))
        return x

#############################################################################################################################

if __name__ == '__main__':
    pass
