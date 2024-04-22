# Echoai Transformer Block Package

This package provides components for building Transformer blocks, including Multi-Head Attention and FeedForward layers.

## Installation

You can install the package using pip:

```bash
pip install echoai-transformer-block
```

Usage
Here's an example of how to use the `Block` class from this package:
```bash
import torch
from echoai_transformer_block import Block

# Define parameters
n_heads = 8
n_embed = 512
block_size = 16
dropout = 0.1
expand = 4

# Create a Block instance
block = Block(n_heads, n_embed, block_size, dropout, expand)

# Example input tensor
input_tensor = torch.randn(1, 16, 512)

# Forward pass through the Block
output_tensor = block(input_tensor)

print("Output shape:", output_tensor.shape)
```
### Components
`Block`
```bash
from echoai_transformer_block import Block
```
The `Block` class represents a Transformer block containing a Multi-Head Attention layer followed by a FeedForward layer.

`MultiAttentionHead`
```bash
from echoai_transformer_block import MultiAttentionHead
```
The `MultiAttentionHead` class represents the Multi-Head Attention mechanism used in Transformers.

`FeedForward`
```bash
from echoai_transformer_block import FeedForward
```
The `FeedForward` class represents the FeedForward module used in Transformer encoder layers.


Requirements
Python 3.7+
PyTorch

License
This project is licensed under the MIT License - see the LICENSE file for details.

This README file includes:
- The updated package name "Echoai Transformer Block".
- Installation instructions for the package.
- An example of how to use the `Block` class.
- Descriptions and import statements for each component (`Block`, `MultiAttentionHead`, `FeedForward`).
- Requirements section listing the required Python version and PyTorch.
- Mention of the license for the package.

