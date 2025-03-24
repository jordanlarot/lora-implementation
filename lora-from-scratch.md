# Implementing LoRA from Scratch in PyTorch - Intuition & Code

Last Updated: 2025-03-24

## High-Level Overview

Low-Rank Adaptation (LoRA) is an approach to efficiently fine-tune a model by freezing the original, pre-trained weights and injecting additional trainable parameters. 

Low Intrinsic Dimension Hypothesis suggests that only a small subset of weight updates matter, so fine-tuning can be done efficiently with fewer parameters. If it is true that some parameters in a network are low-rank (more correlated), it means it can be compressed. In other words, LoRA assumes full-rank updates aren’t needed to adapt large models. It uses low-rank matrices to approximate what the full update would have been:

$$ \Delta W = BA $$ 

## What the Paper did
In [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), the authors inject trainable LoRA parameters in the attention layer. Specifically, in the Query ($ W_Q$) and Value ($ W_V$) parameters. In practice, LoRA can be implemented in any layer of a neural network. Compared to other methods, LoRA has the benefit of lower memory cost, efficient task switching, and no additional inference latency. Note that LoRA is *not being added as a new layer in a neural network*. Rather, it is being introduced as additional trainable parameters in existing layers. 

## Coding Implementation in PyTorch

### LoRA PyTorch Layer
```python
import torch
import torch.nn as nn

class LoRA(nn.Module):
    def __init__(self, embed_dim, rank):
        super().__init__()
        self.rank = rank 

        # Low-rank trainable matrices
        self.A = nn.Parameter(torch.randn(embed_dim, rank) * 0.01) # Matrix A is initialized from a Gaussian distribution; 0.01 as standard deviation
        self.B = nn.Parameter(torch.zeros(rank, embed_dim)) # Matrix B is initialized to 0 

    def forward(self, X):
        # LoRA update: matrix multiplication
        X = X @ self.A @ self.B

        return X 
```

Here we define a `LoRA` class in PyTorch. It has two parameters *embedding dimension* $ d_\text{model} $ and *rank* $ r $. The embedding dimension represents the number of features for a given sequence (e.g. semantic, syntax, etc.). Rank is the dimensional capacity of a matrix to represent information. Lower rank means more dependence, while higher rank means more independence. 

We use `nn.Parameter()` to register these matrices as trainable weights, meaning that the numbers can be updated during backpropagation. Using `torch.tensor()` will not work as `PyTorch` would not treat these as trainable weights. $ A $ will be of shape `{embedding dimension x rank}` while $ B $ will be of shape `{rank x embedding dimension}`. Moreover, in the paper, $ A $ is initialized by a Gaussian distribution and $ B $ is initialized to 0. From there, we perform the following matrix multiplication: 

$$ Output_\text{LoRA} = X \quad @ \quad A \quad @ \quad B $$

$ X $ will be of shape `{batch size x sequence length x embedding dimension}`. 
Performing $ X @ A  $ projects down to $ r $, whereas $ X @ A @ B $ projects it back to $ d_\text{model} $. In other words, we are compressing the input using $ A $, and then reconstructing it back using $ B $.

At first glance, projecting down to a lower dimension seems counterintuitive. Traditional widsom suggests to increase the dimensionality of the features, so we can learn more nuanced patterns. 

However, LoRA wants to project into a lower-dimensional space to efficiently adapt the model. 

> Simply put, the goal is not to learn the features from scratch but to slightly modify an already-trained model

Just like many ML problems can be solved using only a few meaningful input features, LoRA assumes that model adaptation can be achieved by learning updates in a low-dimensional subspace — a small set of directions that account for most of the change. It is saying "we don't need to move in all directions, just a few meaningful ones". If a small number of dimensions can explain most of the effect, then we don’t need to model everything — just that small subspace.

### Injecting LoRA in Self-Attention

```python
import torch 
import torch.nn as nn

class LoRAAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, rank):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rank = rank 
        self.head_dim = embed_dim // num_heads

        # Original frozen weights matrices (Wq, Wk, Wv, Wo)
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wo = nn.Linear(embed_dim, embed_dim, bias=False)

        # LoRA low-rank adapters for Wq and Wv; don't modify Wk, Wo
        self.lora_q = LoRA(embed_dim, rank)
        self.lora_v = LoRA(embed_dim, rank)
        
    def forward(self, X):

        # Compute query, key, value 
        Q = self.Wq(X) + self.lora_q(X) # LoRa for Q 
        K = self.Wk(X) # no LoRA for K 
        V = self.Wv(X) + self.lora_v(X) # LoRa for V 

        # Compute scaled dot-product attention 
        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = scores.softmax(dim=-1)
        attn_output = attn_weights @ V 

        # Apply output project (Wo)
        output = self.Wo(attn_output)

        return output
```

As mentioned earlier, we implement LoRA in the self-attention layer. We inject the LoRA updates to $ W_Q $ and $ W_V $. *Note that we haven't frozen the parameters yet; we will create a separate function for that later.*

To preserve the original output shape, the update weights are added to the original weights as follows. 

$$ W = W_0 + \Delta W $$

### Defining the Transformer Block

```python
import torch 
import torch.nn as nn

class LoRAEncoderTransformerBlock(nn.Module):
    
    def __init__(self, embed_dim, num_heads, ff_dim, rank):
        super().__init__()

        # Define layers 
        self.lora_attention = LoRAAttention(embed_dim, num_heads, rank)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(), 
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, X):

        # Multi-Head Attention
        attn_output = self.lora_attention(X)

        # Add & Norm
        X = self.norm1(X + attn_output)

        # Feed-forward
        ff_output = self.ff(X)

        # Add & Norm
        X = self.norm2(X + ff_output)

        return X
```
This Encoder Transformer Block uses the `LoRAAttention` we created in the previous section instead of Multi-Head Attention. 

### LoRA Transformer

```python
import torch 
import torch.nn as nn

class LoRATransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len, rank):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim) # input embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim)) # Positional encoding
        self.layers = nn.ModuleList([
            LoRAEncoderTransformerBlock(embed_dim, num_heads, ff_dim, rank) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(embed_dim, vocab_size) # final output

    def forward(self, X):
        # Apply positional encoding to input embeddings
        X = self.embed(X) + self.pos_embed[:, :X.shape[1], :]

        # Nx blocks 
        for layer in self.layers:
            X = layer(X)

        # Project to vocab size for classification
        X = self.output_layer(X)

        return X 
```

We implement the entire Transformer using the `LoRAEncoderTransformerBlock` class we created in the previous section. Essentially, we are composing all the layers together. 


### Freeze pre-trained, weights

```python
def freeze_original_weights(model):
    """
    Freezes all non-LoRA parameters so only LoRA parameters are trainable.
    """
    for name, param in model.named_parameters():
        if "lora" not in name:  # Only LoRA parameters should be trainable
            param.requires_grad = False
```
We implement a function to freeze the original weights. This function freezes all parameters except those with `lora` in their name, ensuring only the LoRA adapters are updated during training. When we run this function, only `lora_q` and `lora_v` will be able to be updated during backpropagation.  

## Train Model

```python
import torch
import torch.optim as optim 
import torch.nn as nn

# Model hyperparameters 
vocab_size = 1000
embed_dim = 128
num_heads = 4
ff_dim = 256 
num_layers = 2
max_len = 20
rank = 4 # LoRA rank 

# Initialize LoRA transformer
model = LoRATransformer(vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, num_layers=num_layers, max_len=max_len, rank=rank)

# Freeze original Transformer weights (only train LoRA)
freeze_original_weights(model)

# Loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# Dummy dataset (random tokens)
X_train = torch.randint(0, vocab_size, (32, max_len))  # Batch of 32
y_train = torch.randint(0, vocab_size, (32, max_len))  # Target

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)  # Forward pass
    loss = criterion(outputs.view(-1, vocab_size), y_train.view(-1))  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update only LoRA parameters
```
It is time to train our model. We initialize the model with random input tensors to simulate a training run. From there, we freeze the original Transformer weights and define our loss function and optimizer. Finally, we run the training loop. 

In practice, you would load in the weights of the pre-trained model from a checkpoint and train it on a language dataset. You will also need to apply softmax to convert logits into probabilities, then map the predicted vocab indices back to words.

## Wrapping Up 
LoRA is a parameter-efficient technique for adapting large models. It assumes that the necessary weight updates live in a low-dimensional subspace, allowing us to fine-tune only a small number of parameters. This reduces memory usage and compute cost while preserving the integrity of the original pre-trained model.
