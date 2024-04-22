import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    """
    Multi-head self-attention mechanism for Transformer.
    
    Args:
        head_size (int): Size of each attention head.
        n_embed (int): Size of the input embedding.
        block_size (int): Block size for the attention mechanism.
    """
    def __init__(self, head_size, n_embed, block_size):
        super().__init__()

        self.head_size = head_size
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        # Creating a lower triangular matrix for masking future positions
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x, training=True):
        """
        Forward pass of the multi-head self-attention mechanism.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C), where
                              B is the batch size, T is the sequence length,
                              and C is the number of input channels.
            training (bool): Whether the model is in training mode or not.

        Returns:
            torch.Tensor: Output tensor after self-attention, of shape (B, T, h).
        """
        B, T, C = x.shape

        # Query
        q = self.query(x)  # (B, T, h)
        # Key
        k = self.key(x)    # (B, T, h)

        # Calculate the attention weights
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, h) @ (B, h, T) ---> (B, T, T)
        # Masking future positions
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        # Softmax
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        
        # Value
        v = self.value(x)  # (B, T, h)
        # Weighted sum
        out = wei @ v  # (B, T, T) @ (B, T, h)  ---> (B, T, h)
        
        return out

#######################################################################################################
class MultiAttentionHead(nn.Module):
    """
    Multi-Head Attention mechanism for Transformers.
    
    Args:
        n_heads (int): Number of attention heads.
        n_embed (int): Size of the input embedding.
        block_size (int): Block size for the attention mechanism.
        dropout (float): Dropout probability.
    """
    def __init__(self, n_heads, n_embed, block_size, dropout):
        super(MultiAttentionHead, self).__init__()
        self.head_size = n_embed // n_heads
        self.heads = nn.ModuleList([Head(self.head_size, n_embed, block_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * self.head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the Multi-Head Attention mechanism.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C), where
                              B is the batch size, T is the sequence length,
                              and C is the number of input channels.

        Returns:
            torch.Tensor: Output tensor after multi-head attention, of shape (B, T, C).
        """
        # Process each attention head
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Project to original embedding size
        out = self.proj(self.dropout(out))
        return out