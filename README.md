## The Hackathon Challenge

### What You're Optimizing
You're given a pre-built neural network that's **intentionally slow**. Your job is to make it fast without breaking its accuracy. The model is a complex ensemble of 4 different sequence models (Mamba2, RetNet, Hawk, and xLSTM) that process streaming data for 50+ different "symbols" (think stock tickers).

### Your Goal

You need to balance 3 different metrics
- **Latency**: Minimize latency so you can adapt to market conditions
- **Accuracy**: Keep a a small MSE so the model quality is still good
- **Throughput**: Handle as many requests/second as possible

### The Competition Flow
1. You receive a stream of inference requests with 32 features each
2. Each request belongs to a specific symbol (SYM_001, SYM_002, etc.)
3. The model maintains separate state for each symbol (like memory)
4. You predict a single value per request
5. The server scores you in real-time based on speed and accuracy

## Participant Guide

### What You Need to Change

The main optimization target is the `process_batch()` method in `example_model.py`. This method currently processes requests one-by-one, which is inefficient. Your job is to make it faster through:

### 1. **PyTorch-Level Optimizations** (Easier)
- Use `torch.compile()` to JIT-compile the model
- Implement mixed precision (fp16/bf16) computation

### 2. **Algorithmic Optimizations** (Medium)
- Implement dynamic batching strategies

### 3. **Custom GPU Kernels** (Advanced)
The model has several expensive operations that are perfect targets for custom kernels:

**Key Bottlenecks to Target:**
- **Mamba2 SSM Updates** (`client/model/mamba2.py:90-91`): 4D tensor operations with broadcasting
- **RetNet Rotary Embeddings** (`client/model/retnet.py:72`): Outer products and position encoding
- **xLSTM Cell Updates** (`client/model/xlstm.py:59-62`): Complex gated state updates
- **Causal Convolutions**: Depthwise convs with state management

All the above are just suggestions, please be creative, try things out, profile and find the important bottlenecks
