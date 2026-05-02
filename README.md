# CoNeTTE: Audio Captioning with Task Embeddings
## ML-CS-GY 6933: Machine Listening (Spring 2026)
**Instructor:** Juan Pablo Bello  
**TA:** Julia Wilkins

---

![Audio Captioning Pipeline](https://img.shields.io/badge/Task-Audio--Captioning-blue?style=for-the-badge)
![Model-CoNeTTE](https://img.shields.io/badge/Model-CoNeTTE-green?style=for-the-badge)
![Framework-PyTorch](https://img.shields.io/badge/Framework-PyTorch-red?style=for-the-badge)

## 🎙️ Project Overview
This project focuses on improving **CoNeTTE**, a state-of-the-art system for **Automated Audio Captioning (AAC)**. The goal of AAC is to generate descriptive, natural language summaries of audio content, bridging the gap between raw signal processing and semantic understanding.

CoNeTTE (CNext-trans) is designed for high efficiency and performance, achieving competitive results with significantly fewer parameters than traditional ensemble-based systems.

---

**Submission Feedback**

Overall Feedback From Bello:

Good project and presentation. As we discussed during Q&A, I suggest extending your modifications to be a network-based fusion of encoders, and recreating the decoding/summarization pipeline of the second SOTA model you discussed. Whether you change CLAP for another pre-trained model would IMO afford less opportunities to learn new techniques. Although, if you do decide to test a change of CLAP for SLAP, perhaps you can just run some small test of music captioning. Good luck

Score: 10 / 10 - A

## 🏗️ Core Architecture: CNext-Trans
The model follows an encoder-decoder framework optimized for audio signals:

- **Encoder (ConvNeXt):** Adapted from computer vision, this robust convolutional backbone extracts deep hierarchical features from audio spectrograms.
- **Encoder (BEATS):** Adding a second encoder with a conformer.

- **Decoder (Transformer):** A vanilla Transformer decoder, trained from scratch, translates audio features into coherent natural language.
- **Efficiency:** CoNeTTE utilizes **4–40× fewer parameters** than competing systems without sacrificing caption quality.

---

## 🚀 Key Contributions & Findings

### 1. AudioCaps Bias Analysis
The authors identified that **AudioCaps (AC)** files are drawn entirely from the AudioSet training set. This means many pretrained encoders have "seen" the test data during their own training phases.
- **Impact:** This leak causes a modest **~5% SPIDEr score inflation**.
- **Takeaway:** Benchmarking must account for pre-training data overlap to accurately assess generalization.

### 2. The Multi-Dataset Challenge
Naïve data mixing (combining AudioCaps, Clotho, MACS, and WavCaps) surprisingly **hurt performance**.
- **The Cause:** Each dataset has distinct writing styles, vocabularies, and sound event emphasis.
- **The Solution:** **Task Embeddings (TE)**.

### 3. Task Embeddings (TE)
By prepending a dataset-specific token (e.g., `<bos_ac>`, `<bos_cl>`) to the decoder:
- **Style Steering:** The model learns to mimic specific dataset styles (sentence length, word choice).
- **Content Shift:** The TEs guide the model on which sound events to prioritize based on the target dataset's characteristics.
- **Performance:** This recovered performance loss from data mixing and significantly improved cross-dataset generalization.

---

## 📊 Results Summary
CoNeTTE remains highly competitive with much larger models:
- **AudioCaps:** 44.1% SPIDEr
- **Clotho:** 30.5% SPIDEr

---

## 🛠️ Implementation Plan (Work in Progress)
> [!NOTE]  
> This README and the implementation plan will be updated regularly as the project progresses.

Our work aims to extend and improve this model for the Spring 2026 Machine Listening final project.

### Planned Enhancements
- [ ] Reproduce baseline CoNeTTE results (ConvNeXt encoder + Transformer decoder + beam search) as our control setup.
- [ ] Evaluate upgraded/switchable encoders, starting with **AST (Audio Spectrogram Transformer)** and additional candidates such as EfficientNet-style audio backbones.
- [ ] Replace or complement beam search with **nucleus sampling (top-p)** to generate diverse caption candidates.
- [ ] Build a multi-candidate decoding pipeline:
  - generate multiple captions with nucleus sampling,
  - rank candidates with automatic metrics and semantic similarity,
  - summarize/select final output with an LLM.
- [ ] Integrate a state-of-the-art LLM as a caption refinement/reranking module (e.g., **GPT-5.4**, **Gemini-family models**), then compare against non-LLM baselines.
- [ ] Benchmark open-source LLM alternatives for the same refinement stage (e.g., **Qwen** and **Gemma** families) to study quality/cost trade-offs.
- [x] Prototype **multi-encoder fusion** (ConvNeXt + BEATs) to combine complementary acoustic representations into a richer decoder input.
  - **Important Note:** The original single-encoder architecture is still available in `Model/model/modeling_beats_conformer_bart.py` for running A/B baseline tests. To use the new multi-encoder fusion architecture,  training scripts need to be updated to import and instantiate the `FusedEncodersConformerBartSeq2SeqForCaptioning` class from `Model/model/modeling_fused_conformer_bart.py`.
  - **Fusion Strategy:** We implemented **Early Feature Fusion**. Both BEATs and ConvNeXt process the input parallelly, and their outputs are temporally aligned and concatenated *before* being passed to the Conformer post-encoder. We did *not* implement late fusion (where BEATs goes through the Conformer first and is combined with ConvNeXt later).
  - **Abhijeet:::: Adding AST (Audio Spectrogram Transformer):** 
  I think we have two options: build directly upon `Model/model/modeling_fused_conformer_bart.py`, or leave it untouched and create another one that will do all three. That way we can experiment with baseline, two encoders, and three encoders. If we go with the first option, we should do this: initialize the AST wrapper in `__init__`, increase the `fused_dim` in `self.fusion_mlp` to account for the AST embedding size, and inside `encode_audio()`: extract AST features, use `F.interpolate` to match the BEATs `target_seq_len` (exactly as done for ConvNeXt), and simply append the AST features to the `torch.cat([beats, convnext, ast], dim=-1)` list!
- [ ] Investigate fine-tuning strategies for cross-domain adaptation under single-dataset and multi-dataset training.

### Architecture Updates (Dual-Encoder Fusion)
During our recent development session, we implemented the dual-encoder fusion without altering the baseline. Here is a summary of the structural additions:
- **`Model/ConvNext.py`**: Added a lightweight wrapper to pull the ConvNeXt model directly from Hugging Face `transformers`.
- **`Model/model/modeling_fused_conformer_bart.py`**: Created the new dual-encoder architecture that fuses BEATs and ConvNeXt via 1D temporal interpolation and MLP aggregation.
- **`Model/model/ConvNext test files/test_fusion.py`**: Created a standalone test script to verify tensor shapes and ensure the end-to-end forward pass and text generation pipeline work mathematically.
- *Note: Original baseline files (`modeling_beats_conformer_bart.py`, `BEATs.py`, `backbone.py`, etc.) were left completely untouched to ensure safe A/B testing.*

### Baseline-to-Advanced Experiment Tracks
1. **Encoder Track:** ConvNeXt vs AST vs fused encoders.
2. **Decoder Track:** Beam search vs nucleus sampling vs hybrid decoding.
3. **LLM Track:** No-LLM postprocessing vs proprietary LLM vs open-source LLM.
4. **Generalization Track:** In-domain and cross-dataset evaluation across AudioCaps/Clotho(+).

### Setup & Requirements
```bash
# Clone the repository
git clone https://github.com/robbietylman/Automatic-Audio-Captioning-ML-2026.git
cd Automatic-Audio-Captioning-ML-2026

# Install dependencies
pip install -r requirements.txt
```

---

---

## 🏆 Winning Model

Our final system integrates multiple state-of-the-art components to achieve superior audio-to-text alignment and caption fluency.

```mermaid
graph TD
    subgraph Model ["Architecture: Multi-Encoder Fusion + Conformer Refinement"]
        Audio[Audio Input] --> Mel(Log Mel Spectrogram)
        
        subgraph Encoders ["Extraction (Ensemble)"]
            Mel --> BEATs[BEATs Encoder]
            Mel --> ConvNeXt[ConvNeXt Encoder]
        end
        
        BEATs -- "Temporal Dependencies" --> Concat
        ConvNeXt -- "Local Texture + Timbre" --> Concat
        
        Concat[Concatenation] --> Conformer[Conformer Post-Encoder]
        
        Conformer -- "Refine & Align Fused Features" --> Project[Projection Layer]
        Project -- "Align Language Space" --> B_Enc[BART Encoder]
        B_Enc -- "Self-Attention" --> B_Dec[BART Decoder]
        
        subgraph Generation ["Nucleus Sampling & Refinement"]
            B_Dec -- "Nucleus Sampling" --> Cands[64 Candidate Captions]
            Cands --> CLAP{CLAP Scoring}
            CLAP -- "Similarity Ranking" --> Rerank[Hybrid Reranking]
            Rerank --> LLM[LLM Summarization]
            LLM --> Final((Final Caption))
        end
    end

    subgraph Eval ["Evaluation: FENSE Metric"]
        Final --> Vec[Text Vectors]
        GT[Ground Truth] --> Vec
        Vec --> Cos[Cosine Similarity]
        Final --> Fluency[Fluency LM Classifier]
        Cos & Fluency --> FENSE[FENSE Score]
    end

    style Model fill:#f0f4ff,stroke:#333,stroke-width:2px
    style Eval fill:#fff0f0,stroke:#333,stroke-width:2px
    style Final fill:#e1f7d5,stroke:#2e7d32,stroke-width:3px
```

### Model Details
- **Audio Processing:** Raw audio is converted into **log mel spectrograms**.
- **Encoder Fusion & Refinement:** 
    - **Primary Encoders (BEATs, ConvNeXt):** Multiple pretrained encoders extract different aspects of sound.
    - **BEATs (Transformer):** Captures long-range temporal dependencies.
    - **ConvNeXt (CNN):** Focuses on local texture and timbre patterns.
    - **Structure:** Each encoder turns the spectrogram into a sequence of feature vectors, which are then combined using **concatenation** for a richer representation than any single encoder.
    - **Conformer Post-Encoder:** Acts as a refinement stage that combines local (convolution) and global (attention) context to refine and align the fused audio features, making them more structured and informative for the decoder.
- **Language Alignment:** Outputs are projected into **BART's language space** to match expected input dimensions and align audio features with language embeddings.
- **Decoding & Generation:**
    - **BART Encoder:** Applies self-attention to build a global understanding of the audio.
    - **BART Decoder:** Uses cross-attention to previously generated tokens for next-token prediction.
    - **Nucleus Sampling:** Generates **64 diverse candidate captions**.
- **Refinement Pipeline:**
    - **CLAP Filtering:** Uses CLAP to compare each caption to the original audio, scoring similarity and discarding weak candidates.
    - **Hybrid Reranking:** Applied to consolidate the best candidates.
    - **LLM Summary:** A final LLM pass summarizes the top results into a single, high-quality caption.

### Tensor Flow (Dual-Encoder Fusion)
To understand how the early fusion is implemented, here is the tensor journey during a forward pass. You can verify this yourself at any time by running `python "Model/model/ConvNext test files/test_fusion.py"` (which uses a mock 1-second audio clip with a batch size of 2):
```text
[DEBUG] BEATs output shape: torch.Size([2, 48, 768])
[DEBUG] ConvNeXt raw output shape: torch.Size([2, 12, 768])
[DEBUG] ConvNeXt interpolated shape: torch.Size([2, 48, 768])
[DEBUG] Fused (Concatenated) shape: torch.Size([2, 48, 1536])
[DEBUG] Post-Aggregation MLP shape: torch.Size([2, 48, 768])
```
- **BEATs** outputs 48 time tokens.
- **ConvNeXt** outputs 12 time tokens due to different spatial downsampling.
- **Interpolation:** We apply 1D linear interpolation across the time axis to stretch ConvNeXt's 12 tokens to 48 tokens.
- **Concatenation:** We concatenate the two sequences along the feature dimension (768 + 768 = 1536).
- **Aggregation MLP:** A Jung-style aggregation block (`LayerNorm -> Linear -> GELU -> Linear`) compresses the 1536-dimensional vector back to 768 dimensions for the Conformer post-encoder.

### Evaluation: FENSE
To ensure accuracy and readability, we utilize the **FENSE** metric:
- **Semantic Accuracy:** Converts text into vectors and determines **cosine similarity** for each vector against ground truth references.
- **Fluency Metric:** Measures fluency using a trained **LM or classifier** that assigns higher scores to sentences resembling natural human language and lower scores to grammatically incorrect or awkward ones.

---

## 📚 References
> **Paper:** [CoNeTTE: Audio Captioning with Task Embeddings](https://arxiv.org/abs/2309.00454)  
> **Original Implementation:** [Labbé et al. (2023)](https://github.com/paullabbe/CoNeTTE)

---
*Created for the NYU Tandon School of Engineering - CS-GY 6933 Machine Listening.*
