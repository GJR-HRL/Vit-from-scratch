# Vision Transformer From Scratch on MNIST  

## Why This Project Exists
CNNs solve MNIST easily.

So instead of using convolutional, I asked:
> Can an image be understood using **only attention**?

This project implements a **minimal Vision Transformer (ViT)** from scratch in PyTorch and trains it on MNIST.

**Validation Accuracy: 96.31%**


<img width="952" height="390" alt="5c8da1c3-d0f0-484d-973d-ed299b54ecf1" src="https://github.com/user-attachments/assets/e6a435b1-2ed3-4dc9-a4dd-52fbd6420434" />

# Real Questions

If we remove convolution:

- How does the model know where things are?
- How do patches communicate?
- Who is responsible for summarizing the image?
- Why does a CLS token stabilize learning?

This repository answers those from first principles.


Run the below script to install the dependencies
```
pip install -r requirements.txt
```

Run ViT_from_scratch.ipynb Notebook on collab or Locally

DataSet : Installs Mnist Dataset through TorchVision inside data folder Size (around 64 mb)

# Model Configuration

```python
batch_size = 64
num_classes = 10
num_channels = 1
img_size = 28
patch_size = 7
patch_num = 16

embedding_dim = 16
num_attention_heads = 4
num_transformer_block = 4
feed_forward_neural_network_nodes = 64

learning_rate = 0.001
epochs = 5
```

### Why Patch Size = 7?

28 / 7 = 4  

So the image becomes:

```
4 x 4 grid → 16 patches
```

Sequence length becomes:

```
16 patches + 1 CLS = 17 tokens
```

# Architecture Overview

```
28x28 Image
     │
     ▼
Conv2D Patch Embedding
     │
     ▼
16 Patch Tokens (16-dim each)
     │
     ▼
Add CLS Token
     │
     ▼
Add Learnable Positional Embeddings
     │
     ▼
4 × Transformer Encoder Blocks
     │
     ▼
Extract CLS Token
     │
     ▼
MLP Head
     │
     ▼
Digit Prediction (0–9)
```

# Step 1 — Patch Embedding (Using Conv2D)

Instead of manually slicing and flattening patches, I used:

```python
Conv2d(
    in_channels=1,
    out_channels=embedding_dim,
    kernel_size=patch_size,
    stride=patch_size
)
```

This does two things at once:

1. Cuts image into non-overlapping 7×7 patches  
2. Projects each patch into a 16-dimensional vector  

Output shape:

```
Input:  (B, 1, 28, 28)
Output: (B, 16, 4, 4)
Reshape → (B, 16 patches, 16 dim)
```

Conv2D here is not used for spatial feature extraction —  
it is used as a **structured linear projection layer**.

---

# Step 2 — The CLS Token

If we mean-pool patch tokens:

```
logits = Linear(mean(patches))
```

Every patch tries to store the entire image summary.

That causes:
- Redundant representations
- Noisy gradients
- Competition between tokens

Instead, we introduce a **CLS token**.

```
[CLS] P1 P2 P3 ... P16
```

Only CLS connects to the loss.

During backprop:

- CLS learns to ask useful questions.
- Patches learn to provide useful information.
- Responsibility is clearly separated.

This is why training is stable even with embedding_dim = 16.

> Attention enables communication.  
> Loss gives responsibility to CLS.


# Step 3 — Positional Embeddings

Common misconception:

❌ “This vector represents position 3.”  
✅ “This vector is stored at index 3.”

We compute:

```python
x[i] = patch_embedding[i] + pos_embedding[i]
```

Each patch already has an **address** (its index in the sequence).

The positional embedding is a **bias vector attached to that address**.

It allows the model to learn:

> "Tokens at index 2 behave differently than tokens at index 14."

Since MNIST input size is fixed:

- Patch count is fixed
- Index always corresponds to the same spatial region

So learnable positional embeddings make more sense than sinusoidal ones here.

Why not use a Fixed position embeddings like 0001 , 0002 , 0003 ? 
- Learnable embeddings allow the model to discover the most effective spatial representations directly from the training data
- Negligible Parameter Cost: The additional parameters required for learnable embeddings are minimal compared to the millions of weights in the Transformer's self-attention and MLP layers. For a standard
image with patches, you only need 197 additional vectors, which is "virtually nothing" in the context of the total model size.
- Empirical studies, including those in the original ViT paper, showed that learnable embeddings often perform slightly better than or equal to fixed sinusoidal ones for vision tasks


# Transformer Encoder Block

Each block contains:

```
Input
  │
  ▼
LayerNorm
  │
  ▼
Multi-Head Self Attention (4 heads)
  │
  ▼
Residual Add
  │
  ▼
LayerNorm
  │
  ▼
Feed Forward Network (16 → 64 → 16)
  │
  ▼
Residual Add
```

Multi-head attention implemented using:

```python
nn.MultiheadAttention
```

Embedding dimension = 16  
Heads = 4  
Head dimension = 4  

Even at this tiny scale, attention captures:
- Global structure

Attention learns relationships like:

- Top stroke interacts with bottom curve
- Left boundary relates to right boundary
- Middle pixels determine class identity

Spatial structure **emerges from communication**.


Classification happens entirely through the CLS token.


# Design Constraints (Intentional)

- Small embedding dimension (16)
- Only 4 transformer blocks
- Only 5 epochs
- No data augmentation



# What Makes This Project Interesting

1. Extremely small ViT (lightweight experiment)
2. Clear gradient responsibility via CLS
3. Learnable positional bias (not sinusoidal)


# Final Result

Validation Accuracy: **96.31%**  
Epochs: 5  
Architecture: Minimal Vision Transformer  

# Possible Extensions

- Replace CLS with mean pooling → compare stability
- Visualize attention maps
- Increase embedding dimension → observe scaling
- Train on CIFAR-10 → analyze limitations
- Compare Conv-based patching vs linear patching


# Closing Thought

This project was about understanding:

- How attention enables communication
- How loss defines responsibility for learning
- How positional embeddings inject structured bias
- How Transformers behave in low-dimensional regimes (mostly can Overfit)