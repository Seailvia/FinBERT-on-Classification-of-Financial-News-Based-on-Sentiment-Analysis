# Financial News Sentiment Analysis with FinBERT

This project focuses on classifying financial news sentiment using the FinBERT model, a domain-adapted BERT model, and compares its performance with traditional machine learning methods. The implementation includes a lightweight fine-tuning strategy using LoRA to adapt FinBERT to both structured financial news and real-time, informal financial discourse such as Twitter tweets.

## Project Overview

Financial sentiment analysis plays a critical role in investment decision-making. This project evaluates the performance of FinBERT, a pre-trained language model specialized for financial text, against traditional machine learning methods (Logistic Regression, SVM, Decision Tree) and deep learning approaches (DAN) for financial sentiment classification. 

A two-phase fine-tuning strategy is introduced: first pre-training on the Financial PhraseBank dataset, then adapting to the Twitter Financial News Sentiment dataset. The lightweight fine-tuning framework Low Rank Adaptation (LoRA) is used to make the model suitable for real-time financial NLP tasks.

## Key Features

- **Domain-Adapted Model**: FinBERT, a BERT-based model fine-tuned for financial text analysis
- **Lightweight Fine-Tuning**: Implementation of LoRA for efficient adaptation to new datasets
- **Comprehensive Evaluation**: Comparison with traditional machine learning and deep learning methods
- **Real-Time Application**: Adaptation to informal financial discourse such as Twitter tweets
- **Open-Source Implementation**: Ready-to-deploy solution for streaming data pipelines

## Datasets

The project utilizes two main datasets:

1. **Financial PhraseBank Dataset**:
   - 4,840 English financial news sentences
   - Derived from news articles related to listed companies in OMX Helsinki
   - Annotated by 16 financial specialists into positive, neutral, and negative sentiments
   - Multiple versions based on agreement strength: 100%, >75%, >66%, >50% 

2. **Twitter {insert\_element\_0\_}Financial News Sentiment Dataset**:
   - 11,932 English-language finance-related tweets
   - Labeled as Bearish (negative), Bullish (positive), or Neutral
   - Represents real-time, unstructured financial information from social media 

## Methodolog{insert\_element\_1\_}y

### Model Architecture

- **FinBERT**: A domain-adapted BERT model fine-tuned on financial corpora including earnings reports, regulatory filings, and financial news 
- **LoRA (Low-{insert\_element\_2\_}Rank Adaptation)**: Lightweight fine-tuning method that freezes pre-trained weights and inserts trainable low-rank matrices
  - \( W = W_0 + \Delta W \), where \( \Delta W = BA \)
  - Applied to query and value projection layers in Transformer blocks
  - Configured with rank \( r=8 \), scaling factor \( \alpha=32 \), and dropout rate 0.1 

### Training {insert\_element\_3\_}Process

1. **Parameter Configuration**: Set dataset paths, model saving paths, sequence lengths, batch sizes, learning rates, etc. 
2. **Model Ini{insert\_element\_4\_}tialization**: Initialize FinBERT with LoRA adapters for efficient fine-tuning 
3. **Data Prep{insert\_element\_5\_}aration**: Load and preprocess training and validation data, compute class weights 
4. **Model Con{insert\_element\_6\_}struction**: Build the model, set up optimizer and learning rate scheduler 
5. **Model Tra{insert\_element\_7\_}ining**: Train for 30 epochs with AdamW optimizer, evaluate on validation data 
6. **Model Sav{insert\_element\_8\_}ing and Deployment**: Save the best-performing model for sentiment analysis tasks 

## Installati{insert\_element\_9\_}on

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.16+
- Scikit-learn 0.24+
- Pandas 1.3+
- NumPy 1.19+
