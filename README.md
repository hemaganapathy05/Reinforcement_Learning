# 🤖 MiniGPT – TinyStories Text Generation using Deep Learning

<p align="center">

### 🧠 Transformer-Based Language Model Trained from Scratch

A Deep Learning project that builds and trains a **MiniGPT-style Transformer language model** from scratch using the **TinyStories dataset** and generates new stories from a given text prompt.

</p>

---

## 📌 Project Overview

This project implements a small **GPT-style Transformer language model** using **PyTorch**.

The model is trained on the **TinyStories dataset** using next-token prediction. It learns the patterns and structure of children's stories and can generate new text based on a user-provided prompt.

Unlike using a pre-trained GPT model directly, the Transformer architecture in this project is **initialized and trained from scratch**.

### 🎯 Main Objective

The main objective of this project is to understand and implement the core concepts behind modern **Generative AI and Large Language Models (LLMs)**, including:

* Tokenization
* Word/token embeddings
* Positional embeddings
* Self-attention
* Causal masking
* Transformer blocks
* Next-token prediction
* Cross-entropy loss
* AdamW optimization
* Text generation
* Temperature sampling
* Top-k sampling

---

## ✨ Features

* 🧠 MiniGPT Transformer architecture
* 📚 TinyStories dataset
* 🔤 GPT-2 tokenizer
* ⚡ PyTorch implementation
* 🎯 Causal self-attention
* 🔄 Multiple Transformer blocks
* 📉 Cross-entropy training loss
* 🚀 AdamW optimizer
* 🎲 Temperature-based text generation
* 🔝 Top-k sampling
* 💾 Model checkpoint saving
* 🎨 Story generation from custom prompts
* 🖥️ CUDA/GPU support

---

## 🏗️ Model Architecture

The project follows a simplified GPT architecture:

```text
                 Input Text
                     │
                     ▼
              GPT-2 Tokenizer
                     │
                     ▼
               Token IDs
                     │
                     ▼
             Token Embeddings
                     │
                     +
             Positional Embeddings
                     │
                     ▼
        ┌──────────────────────────┐
        │   Transformer Block 1    │
        │                          │
        │ LayerNorm                │
        │      ↓                   │
        │ Causal Self-Attention    │
        │      ↓                   │
        │ Residual Connection      │
        │      ↓                   │
        │ LayerNorm                │
        │      ↓                   │
        │ Feed Forward / MLP       │
        │      ↓                   │
        │ Residual Connection      │
        └──────────────────────────┘
                     │
                    ...
                     │
        ┌──────────────────────────┐
        │   Transformer Block 6    │
        └──────────────────────────┘
                     │
                     ▼
                LayerNorm
                     │
                     ▼
              Language Model Head
                     │
                     ▼
               Token Probabilities
                     │
                     ▼
               Generated Text
```

---

## 📊 Dataset

### TinyStories

The project uses the **TinyStories** dataset, a collection of short synthetic stories designed for training small language models.

For this implementation, a subset of:

```text
50,000 stories
```

is used for training.

The dataset is divided into:

```text
95% → Training Data
5%  → Validation Data
```

Each story is tokenized and converted into token IDs before being used for model training.

---

## 🔤 Tokenization

The project uses the **GPT-2 tokenizer** from Hugging Face.

```python
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
```

The tokenizer converts text into numerical token IDs that can be processed by the neural network.

Example:

```text
Input:
"Once upon a time"

        ↓ Tokenization

Token IDs:
[...]
```

The GPT-2 tokenizer is used only for **tokenization**. The MiniGPT model itself is trained from scratch.

---

## 🧠 Transformer Components

### 1. Token Embedding

Each token is converted into a dense numerical vector.

```python
nn.Embedding(
    vocab_size,
    n_embd
)
```

---

### 2. Positional Embedding

Since Transformers do not inherently understand word order, positional embeddings are added to represent the position of each token.

```python
nn.Embedding(
    block_size,
    n_embd
)
```

---

### 3. Causal Self-Attention

Causal self-attention allows the model to understand relationships between tokens while preventing it from looking at future tokens.

For example:

```text
The cat sat on the ...
```

When predicting the next token, the model can use:

```text
The → cat → sat → on → the
```

but cannot access the future answer.

---

### 4. Feed Forward Network

Each Transformer block contains a Multi-Layer Perceptron (MLP).

```text
Linear
   ↓
GELU
   ↓
Linear
```

---

### 5. Layer Normalization

Layer normalization is applied before the attention and MLP components to improve training stability.

---

### 6. Residual Connections

Residual connections allow information to flow through the Transformer blocks and help with stable deep learning.

---

## ⚙️ Model Configuration

The MiniGPT model uses the following configuration:

| Parameter           |            Value |
| ------------------- | ---------------: |
| Vocabulary          | GPT-2 vocabulary |
| Block Size          |              256 |
| Embedding Dimension |              256 |
| Attention Heads     |                4 |
| Transformer Layers  |                6 |
| Batch Size          |               32 |
| Learning Rate       |             3e-4 |
| Optimizer           |            AdamW |
| Weight Decay        |              0.1 |
| Loss Function       |    Cross Entropy |
| Dataset             |      TinyStories |
| Training Samples    |   50,000 stories |

---

## 🔄 Training Process

The model is trained using **next-token prediction**.

The training process can be represented as:

```text
TinyStories Dataset
        │
        ▼
     Tokenizer
        │
        ▼
   Token Sequences
        │
        ▼
 Training / Validation Split
        │
        ▼
      MiniGPT
        │
        ▼
 Causal Self-Attention
        │
        ▼
 Transformer Blocks
        │
        ▼
   Language Model Head
        │
        ▼
   Predicted Token
        │
        ▼
 Cross-Entropy Loss
        │
        ▼
    Backpropagation
        │
        ▼
      AdamW
        │
        ▼
 Updated Model Weights
```

---

## 📉 Loss Function

The project uses **Cross-Entropy Loss** for next-token prediction.

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    targets.view(-1)
)
```

The objective is to minimize the difference between:

```text
Predicted next token
        vs
Actual next token
```

As training progresses, the model should learn better representations of the training data.

---

## 🚀 Text Generation

After training, the model can generate stories from a custom prompt.

Example:

```text
Once upon a time there was a little girl named Lily.
```

The model predicts the next token repeatedly:

```text
Prompt
  ↓
Predict next token
  ↓
Add token
  ↓
Predict next token
  ↓
Add token
  ↓
...
  ↓
Generated Story
```

---

## 🎲 Temperature Sampling

Temperature controls the randomness of generated text.

```python
logits = logits / temperature
```

A lower temperature generally produces more predictable text.

A higher temperature produces more diverse/random text.

Example:

```text
temperature = 0.5
→ More predictable

temperature = 0.8
→ Balanced

temperature = 1.0
→ More random
```

---

## 🔝 Top-K Sampling

The project also uses Top-K sampling.

```python
top_k = 50
```

Instead of sampling from the entire vocabulary, the model considers the top K most likely tokens.

This helps improve the quality and coherence of generated text.

---

# 🛠️ Technologies Used

| Technology               | Purpose                 |
| ------------------------ | ----------------------- |
| 🐍 Python                | Programming language    |
| 🔥 PyTorch               | Deep Learning framework |
| 🤗 Hugging Face Datasets | Dataset loading         |
| 🤗 Transformers          | GPT-2 tokenizer         |
| 📚 TinyStories           | Training dataset        |
| ⚡ CUDA                   | GPU acceleration        |
| 🧠 Transformer           | Model architecture      |
| 📓 Google Colab          | Development environment |

---

# 📂 Project Structure

```text
MiniGPT-TinyStories/
│
├── README.md
│
├── Untitled17.ipynb
│
├── mini_gpt_tinystories.pt
│
└── mini_gpt_tokenizer/
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── ...
```

> The model checkpoint and tokenizer directory are generated after training.

---

# 💻 How to Run

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/MiniGPT-TinyStories.git
```

Move into the project directory:

```bash
cd MiniGPT-TinyStories
```

---

## 2️⃣ Open the Notebook

Open:

```text
Untitled17.ipynb
```

using **Google Colab** or Jupyter Notebook.

---

## 3️⃣ Enable GPU

For Google Colab:

```text
Runtime
   ↓
Change runtime type
   ↓
Hardware accelerator
   ↓
T4 GPU
```

GPU is strongly recommended for training.

---

## 4️⃣ Install Dependencies

Run:

```bash
pip install datasets transformers tqdm
```

---

## 5️⃣ Load Dataset

The notebook automatically downloads the TinyStories dataset:

```python
from datasets import load_dataset

dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train[:50000]"
)
```

No manual dataset download is required.

---

## 6️⃣ Train the Model

For a quick test, use:

```python
MAX_STEPS = 200
```

Once the complete pipeline works correctly, use:

```python
MAX_STEPS = 5000
```

for longer training.

---

## 7️⃣ Generate Text

After training:

```python
prompt = "Once upon a time there was a little girl named Lily."

generated_story = generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=150,
    temperature=0.8,
    top_k=50
)

print(generated_story)
```

---

# 💾 Model Saving

After training, the model is saved as:

```text
mini_gpt_tinystories.pt
```

The tokenizer is saved in:

```text
mini_gpt_tokenizer/
```

The checkpoint contains the model weights and configuration.

---

# 📈 Expected Output

During training, the notebook displays information such as:

```text
Step   100 | Train Loss ... | Val Loss ...
Step   200 | Train Loss ... | Val Loss ...
Step   300 | Train Loss ... | Val Loss ...
```

After training, the model generates a story from the supplied prompt.

Example:

```text
Prompt:
Once upon a time there was a little girl named Lily.

Generated:
Once upon a time there was a little girl named Lily.
She loved playing in the garden with her little friend...
```

The exact generated output will vary because text generation uses probabilistic sampling.

---

# 🎯 Project Objectives

* Understand Transformer architecture
* Implement causal self-attention
* Understand GPT-style language modeling
* Learn tokenization and embeddings
* Train a language model from scratch
* Understand next-token prediction
* Implement text generation
* Experiment with temperature and Top-K sampling
* Gain practical experience with PyTorch

---

# 🌟 Advantages

* ✅ Implements a GPT-style architecture from scratch
* ✅ Helps understand how Transformers work internally
* ✅ Uses GPU acceleration
* ✅ Uses a real language-model training dataset
* ✅ Supports custom text prompts
* ✅ Demonstrates Generative AI concepts
* ✅ Saves the trained model for later use

---

# ⚠️ Limitations

This is a **small educational language model**, not a production-scale LLM.

Possible limitations include:

* Limited training dataset
* Small model size
* Limited context length
* Generated text may sometimes be repetitive
* Training quality depends on the number of training steps
* CPU training is significantly slower than GPU training

---

# 🔮 Future Enhancements

The project can be extended with:

* [ ] Larger TinyStories dataset
* [ ] More Transformer layers
* [ ] Larger embedding dimension
* [ ] Learning-rate scheduling
* [ ] Mixed-precision training
* [ ] Better evaluation metrics
* [ ] Perplexity calculation
* [ ] Web-based text-generation interface
* [ ] Streamlit deployment
* [ ] Model checkpoint/resume training
* [ ] Fine-tuning on a custom dataset
* [ ] Interactive chatbot interface

---

# 🧪 Learning Outcomes

Through this project, the following concepts are demonstrated:

```text
Python
  ↓
PyTorch
  ↓
Neural Networks
  ↓
Attention Mechanism
  ↓
Transformer Architecture
  ↓
Language Modeling
  ↓
Generative AI
```

---

# 📌 Project Type

**Category:** Deep Learning / NLP / Generative AI

**Task:** Text Generation

**Architecture:** Transformer / GPT-style Model

**Learning Type:** Self-Supervised Learning

**Framework:** PyTorch

**Dataset:** TinyStories

---

# ⭐ Acknowledgements

* **TinyStories Dataset** – for providing the training data.
* **Hugging Face** – for the dataset and tokenizer ecosystem.
* **PyTorch** – for the deep learning framework.
* **Google Colab** – for GPU-based experimentation.

---

# 📜 License

This project is created for **educational and academic purposes**.

You are free to use, modify, and extend the implementation for learning and experimentation.

---

## ⭐ If you found this project useful

Give the repository a ⭐ and feel free to explore, modify, and improve the MiniGPT implementation!

---

### 🔖 Topics

```text
deep-learning
machine-learning
nlp
generative-ai
transformers
gpt
minigpt
pytorch
tinystories
language-model
text-generation
self-supervised-learning
artificial-intelligence
llm
```
