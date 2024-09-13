# Basics, Training and Scaling laws
- Construction Process of Large Language Models
  - Large-scale Pre-training
  - Instruction Tuning and Human Preference
- Scaling Laws
  - KM Scaling Law
  - Chinchilla Scaling Law

# Datasets and Training Libraries
- Common Pre-training Datasets
  - Web Pages
  - Books
  - Wikipedia
  - Code
- Common Fine-tuning Datasets
  - Instruction-following Datasets
  - Human Preference Datasets
- Training Libraries
  - Hugging Face Open Source Community
  - FSDP
  - DeepSpeed
  - Megatron-LM

# Data Preparation
- Data Sources
  - General Text Data (Common Crawl, Wikipedia, News articles, public domain books)
  - Specialized Text Data (Scientific Papers, Legal Documents, Medical Records, Code repos)
- Data Preprocessing
  - Quality Filtering
  - Sensitive Content Filtering
  - Data Deduplication
- Tokenization (Word Segmentation)
  - BPE Tokenization
  - SentencePiece Tokenization
  - WordPiece Tokenization
  - Unigram Tokenization

# Model Architecture
- Transformer Model
  - Input Embedding
  - Multi-head Self-attention Mechanism
  - Feedforward Network Layer
  - Encoder
  - Decoder
- Detailed Configuration
  - Normalization Method
  - Normalization Position
  - Activation Function
  - Position Encoding
  - Attention Mechanism
  - Mixture of Experts Model
- Mainstream Architectures
  - Encoder-Decoder Architecture
  - Causal Decoder Architecture
  - Prefix Decoder Architecture
- Long Context Length
  - Extended Position Encoding
  - Sparse Attention Window
  - Long Text Data
- New Model Architectures
  - State Space Models
  - RWKV
  - X-LSTM

# Model Pre-training
- Pre-training Tasks
  - Language Modeling
  - Masked Language Modeling
  - Mixed Masking
- Optimization Algorithm
  - Batch-based Training
  - Learning Rate
  - Optimizers
- Scalable Training Techniques
  - 3D Parallel Training
  - Zero Redundancy Optimizer
  - Activation Recomputation
  - Mixed Precision Training

# Instruction Fine-tuning
- Instruction Data Construction
  - NLP Task Datasets
  - Daily Dialogue Datasets
  - Synthetic Data
- Instruction Tuning Training Strategies
  - Optimization Settings
  - Data Organization Strategies
- Parameter-efficient Model Tuning
  - Low-rank Adaptation Methods

# Chapter 8: Human Alignments
- Reinforcement Learning Based on Human Feedback
  - Collection of Human Feedback Data
  - Reward Model Training
  - Reinforcement Learning Training
- Non-Reinforcement Learning Alignment Methods
  - Collection of Alignment Data
  - Representative Supervised Alignment Method DPO
  - Other Supervised Alignment Methods

# Decoding and Serving
- Decoding Strategy
  - Greedy Search Strategy
  - Random Sampling Strategy
  - Practical Usage Settings
- Decoding Acceleration Methods
  - Decoding Efficiency Analysis
  - System-level Optimization
  - Decoding Algorithm Optimization
- Low-resource Deployment Strategy
  - Quantization
  - Large Model Quantization Methods
  - Model Pruning
  - Model Distillation

# Prompt Learning
- Basic Prompts
  - Manual Prompt Design
  - Automatic Prompt Optimization
- In-context Learning
- Chain-of-Thought Prompting
  - Basic Forms of Chain-of-Thought Prompting
  - Optimization Strategies for Chain-of-Thought Prompting

# Planning and Tool Use
- Planning Based on Large Language Models
  - Plan Generation
  - Feedback Collection
- Tool Use Based on Large Language Models
  - Structure of Large Language Model Tool Use
  - Structure of Multi-Agent Systems

# Evaluation
- Basic Capability Evaluation
  - Language Generation
  - Knowledge Utilization
  - Complex Reasoning
- Advanced Capability Evaluation
  - Human Alignment
  - Environmental Interaction
  - Tool Use
- Open Comprehensive Evaluation Systems
  - MMLU
  - BIG-Bench
  - HELM
  - C-Eval