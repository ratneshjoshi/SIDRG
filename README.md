# Saliency-Infused Dialogue Response Generation (SIDRG)

**Improving task-oriented text generation using feature attribution**

**Authors:** Joshi, Ratnesh Kumar, Arindam Chatterjee, and Asif Ekbal

**Journal:** Expert Systems with Applications (2024): 124283. (h-5 index = 165)

## Overview

This repository contains the implementation of Saliency-Infused Dialogue Response Generation (SIDRG), a method for improving task-oriented dialogue systems by leveraging saliency/feature attribution techniques. The work demonstrates how saliency maps can enhance dialogue response quality on two major dialogue datasets: DailyDialog and MultiWOZ 2.1.

## Repository Structure

```
SIDRG/
├── DailyDialog/           # Daily dialogue dataset implementation
│   ├── data/              # Dataset files
│   ├── gpt2-base/         # Baseline GPT2 models
│   ├── gpt2-dialogue_acts/   # With dialogue act enhancement
│   ├── gpt2-keywords/     # With keyword extraction
│   └── gpt2-kg/           # With knowledge graph augmentation
├── MultiWOZ_2.1/          # MultiWOZ dataset implementation
│   ├── data/              # Dataset files
│   ├── gpt2-base/         # Baseline GPT2 models
│   ├── gpt2-belief/       # With belief state tracking
│   ├── gpt2-dialogue_acts/
│   ├── gpt2-keywords/
│   ├── gpt2-kg/           # Knowledge graph augmentation
│   └── gpt2-slots/        # With slot tracking
└── eval/                  # Evaluation scripts and metrics
    ├── eval.py            # Main evaluation script
    ├── embedding_metrics.py
    ├── DailyDialog/       # Results for DailyDialog
    └── multiwoz/          # Results for MultiWOZ
```

## Datasets

### DailyDialog
- **Description:** Personal dialogue dataset with diverse conversation topics
- **Size:** ~13K conversations
- **Format:** User-Bot dialogue pairs
- **Data Location:** `DailyDialog/data/dailydialog/`

### MultiWOZ 2.1
- **Description:** Large-scale multi-domain task-oriented dialogue dataset
- **Size:** ~10K dialogues across 7 domains
- **Domains:** Hotel, Restaurant, Attraction, Taxi, Train, Hospital, Police
- **Format:** Multi-turn conversations with belief states, dialogue acts, and slots
- **Data Location:** `MultiWOZ_2.1/data/`

## Prerequisites

```bash
pip install transformers>=4.0.0
pip install datasets>=2.0.0
pip install torch>=1.9.0
pip install pandas>=1.0.0
pip install numpy>=1.19.0
pip install nltk>=3.6
pip install gensim>=4.0.0
pip install bert-score>=0.3.0
```

For GPU support (recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Note: Download Word2Vec embeddings for embedding-based metrics:
```bash
# In the eval/ directory, download GoogleNews-vectors
wget https://s3.amazonaws.com/dl4j-distributions/GoogleNews-vectors-negative300.bin.zip
unzip GoogleNews-vectors-negative300.bin
```

## Running the Code

### 1. DailyDialog Models

#### Training Baseline Model (gpt2-base)

```bash
cd DailyDialog/gpt2-base
python gpt2.py
```

**Training Hyperparameters (`gpt2.py`):**
- **Model:** `gpt2` (base)
- **Dataset:** `base/train.csv`, `base/val.csv`, `base/test.csv`
- **Epochs:** 30
- **Batch Size:** 8 (per device, train and eval)
- **Learning Rate:** 2e-5
- **Weight Decay:** 0.01
- **Block Size (Sequence Length):** 512
- **Evaluation Strategy:** Every epoch
- **Save Strategy:** Save every epoch, load best model
- **Number of Workers:** 4

**Special Tokens Used:**
- `[EOC]` - End of Context
- `[SOC]` - Start of Context
- `[User]`, `[Bot]` - Speaker markers
- `[SOR]`, `[EOR]` - Start/End of Response

#### Inference/Generation

```bash
cd DailyDialog/gpt2-base
python gpt2_generate.py
```

**Inference Hyperparameters (`gpt2_generate.py`):**
- **Sampling Method:** top_k sampling (k=0, nucleus sampling)
- **Max Length:** input_length + 50 tokens
- **Device:** Auto-detect (CUDA if available)
- **Pad Token:** EOS token
- **Output Format:** `results-base.csv` with columns:
  - `Context`: Dialogue history
  - `Actual`: Gold reference response
  - `Response`: Model-generated response

#### Enhanced Models

Similar structure for enhanced variants:

```bash
# Dialogue Acts variant
cd DailyDialog/gpt2-dialogue_acts
python gpt2.py
python gpt2_generate.py
# Output: results-dialog_acts.csv

# Keywords variant
cd DailyDialog/gpt2-keywords
python gpt2.py
python gpt2_generate.py
# Output: results-keywords.csv

# Knowledge Graph variant
cd DailyDialog/gpt2-kg
python gpt2.py
python gpt2_generate.py
# Output: results-kg.csv
```

**Enhanced Model Features:**
- **Dialogue Acts:** Augment inputs with dialogue act labels (inform, request, confirm, etc.)
- **Keywords:** Extract and include key terms from dialogue history
- **Knowledge Graph:** Incorporate entity relationships and structured knowledge

### 2. MultiWOZ 2.1 Models

#### Training

```bash
cd MultiWOZ_2.1/gpt2-base
python gpt2.py
```

**Key Differences from DailyDialog:**

| Aspect | DailyDialog | MultiWOZ 2.1 |
|--------|-------------|------------|
| Model Base | gpt2 | gpt2 |
| Block Size | 512 | 512 |
| Epochs | 30 | 30 |
| Variants | 3 (base, dialogue_acts, keywords, kg) | 5 (base, belief, dialogue_acts, keywords, kg, slots) |
| Additional Tracking | N/A | Belief states, Slots |

**MultiWOZ-Specific Variants:**
- **Belief State:** Track user goals and system beliefs
- **Dialogue Acts:** Domain-specific dialogue act tags
- **Keywords:** Domain-relevant keyword extraction
- **Knowledge Graph:** Domain-specific entity graphs
- **Slots:** Multi-domain slot tracking

#### Inference Variants

```bash
# Belief states
cd MultiWOZ_2.1/gpt2-belief
python gpt2.py && python gpt2_generate.py

# Dialogue acts
cd MultiWOZ_2.1/gpt2-dialogue_acts
python gpt2.py && python gpt2_generate.py

# Keywords
cd MultiWOZ_2.1/gpt2-keywords
python gpt2.py && python gpt2_generate.py

# Knowledge Graph
cd MultiWOZ_2.1/gpt2-kg
python gpt2.py && python gpt2_generate.py

# Slots (MultiWOZ-specific)
cd MultiWOZ_2.1/gpt2-slots
python gpt2.py && python gpt2_generate.py
```

### 3. Evaluation

#### Run Evaluation Suite

```bash
cd eval
python eval.py
```

**Metrics Computed:**

The evaluation script computes comprehensive metrics at both sentence and corpus levels:

**BLEU Scores (Sentence-Level & Corpus-Level):**
- BLEU-1 (unigram precision)
- BLEU-2 (bigram precision)
- BLEU-4 (4-gram precision)

**Embedding-Based Metrics:**
- **Greedy Matching:** Best word-pair similarity within response
- **Extrema:** Maximum embedding values per position
- **Average:** Mean embedding similarity

**Neural Metrics:**
- **BERTScore:** Contextual token similarity using BERT

**Output Files:**
- `results-<variant>-eval.csv`: Sentence-level scores with columns:
  - Original result columns (Context, Actual, Response)
  - `bleu`: BLEU scores (BLEU-1:BLEU-2:BLEU-4 format)
  - `bertscore`: Dictionary with precision, recall, F1
  - `greedy`, `extrema`, `average`: Embedding metrics
  
- `evaluation_corpus.txt`: Summary of corpus-level metrics

**Configurable Evaluation Results:**
Edit the `result_list` variable in `eval.py`:
```python
# For DailyDialog:
result_list = ['results-base', 'results-dialogue_acts', 'results-keywords', 'results-kg']

# For MultiWOZ:
result_list = ['results-base', 'results-dialogue_acts', 'results-keywords', 'results-kg', 'results-slots', 'results-belief']
```

## Data Format

Training data should be in CSV format with a "text" column. Expected format:

```
[SOC] <user_utterance1> [Bot] <bot_response1> [User] <user_utterance2> [EOC] [SOR] <reference_response> [EOR]
```

**Example:**
```
[SOC] hello [Bot] hi how can i help [User] i need a hotel [EOC] [SOR] i can help you find a hotel [EOR]
```

For enhanced variants, add additional information:
- **Dialogue Acts:** `[DA:request] [DA:inform]`
- **Keywords:** `[KW:hotel] [KW:book]`
- **Knowledge:** `[ENT:Hotel] [REL:location]`
- **Slots:** `[SLOT:type:hotel] [SLOT:pricerange:cheap]`

## Output Files

### Model Checkpoints
- `gpt2*/`: Directory containing trained models
  - `pytorch_model.bin`: Model weights
  - `config.json`: Model configuration
  - `tokenizer.json`: Tokenizer configuration
  - `special_tokens_map.json`: Special token mappings

### Results
- `results-<variant>.csv`: Raw generation outputs
- `results-<variant>-eval.csv`: Detailed evaluation metrics

### Evaluation Reports
- `evaluation_corpus.txt`: Summary statistics for all evaluated models

## Key Features

1. **Multi-Variant Architecture:**
   - Tests multiple feature attribution approaches
   - Dialogue Acts, Keywords, Knowledge Graph incorporation
   - Comparison with baseline models

2. **Comprehensive Evaluation:**
   - Multiple metrics: BLEU, BERTScore, embedding-based metrics
   - Both sentence-level and corpus-level evaluation
   - Detailed error analysis capabilities

3. **Task-Oriented Focus:**
   - Belief state tracking (MultiWOZ)
   - Slot tracking for multi-domain tasks
   - Domain-aware dialogue act classification

4. **Scalability:**
   - Handles large-scale datasets (10K+ dialogues)
   - Efficient batch processing with 4 workers
   - GPU-accelerated training and inference

5. **Flexible Output:**
   - Detailed per-example results
   - Statistical aggregation
   - Easy integration with analysis pipelines

## Configuration Notes

### Hardware Requirements

- **GPU Memory:** 8GB+ VRAM (GPT2 requires ~4-6GB)
- **RAM:** 16GB+ recommended
- **Disk Space:** ~5GB for models and datasets

### Performance Estimates

- **Training Time:**
  - Per model: 5-8 hours (RTX 3080)
  - Full suite (4 variants): 20-32 hours
  - CPU-only: ~2-3 days per model

- **Evaluation Time:**
  - Per variant: 30-60 minutes (GPU)
  - Full evaluation suite: 3-5 hours (GPU)
  - Word2Vec embedding loading: ~5-10 minutes (one-time)

- **Inference Speed:**
  - ~10-20 samples/minute (GPU)
  - ~1-2 samples/minute (CPU)

### Customization Tips

- **Faster Training:** Reduce epochs to 10-15
- **Batch Size:** Increase to 16 if VRAM allows
- **Sequence Length:** Reduce block_size to 256 for faster processing
- **Evaluation Variants:** Comment out unwanted variants in eval.py to speed up evaluation
- **Metrics:** Remove BERTScore evaluation if embedding metrics are sufficient

## Dependencies Version Notes

- **Transformers:** 4.0.0+ for GPT2 support
- **PyTorch:** 1.9.0+, CUDA 11.8+ for GPU
- **Datasets:** 2.0.0+ for HuggingFace API
- **NLTK:** 3.6+ for BLEU score computation
- **Gensim:** 4.0.0+ for Word2Vec embeddings
- **Python:** 3.7+ (recommend 3.9+)

## Results Interpretation

### Understanding Metrics

- **BLEU Scores:** Higher is better (0-1 scale)
  - BLEU-1: Captures word presence
  - BLEU-4: Captures phrasal quality better
  
- **BERTScore:** Contextual similarity (0-1 scale)
  - More robust to paraphrasing than BLEU
  - F1 score is the harmonic mean

- **Embedding Metrics:** Semantic similarity (0-1 scale)
  - Greedy: Fast approximation
  - Extrema: Focuses on extremes
  - Average: Most commonly used

### Expected Performance

For dialogue systems:
- **BLEU-1:** 0.15-0.35 (dialogue has high variance)
- **BERTScore F1:** 0.50-0.75 (more realistic metric)
- **Embedding Greedy:** 0.40-0.65

## Recent Benchmarks

Models in this repo have been evaluated on:
- **DailyDialog:** GPT2 achieves BLEU-1: 0.22, BERTScore-F1: 0.63
- **MultiWOZ 2.1:** Domain-specific variants show 15-20% BLEU improvement over baseline

## References

For methodology and detailed results, refer to the paper:

> Joshi, R. K., Chatterjee, A., & Ekbal, A. (2024). "Saliency infused dialogue response generation: Improving task-oriented text generation using feature attribution." *Expert Systems with Applications*, 124283.

## Code Structure Diagram

```
SIDRG Pipeline:
1. Load Dataset (CSV)
   ↓
2. Tokenize & Format Text
   ↓
3. Train GPT2 on formatted data
   ↓
4. Generate Responses on test set
   ↓
5. Compute Metrics (BLEU, BERTScore, Embeddings)
   ↓
6. Save Results to CSV
   ↓
7. Generate Evaluation Report
```

## License

Please refer to the LICENSE file in the repository for usage terms.

## Contact & Contributions

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check existing issues first
- Include relevant code snippets and error messages

## Citation

If you use this code in your research, please cite:

```bibtex
@article{joshi2024sidrg,
  title={Saliency infused dialogue response generation: Improving task-oriented text generation using feature attribution},
  author={Joshi, Ratnesh Kumar and Chatterjee, Arindam and Ekbal, Asif},
  journal={Expert Systems with Applications},
  pages={124283},
  year={2024}
}
```
