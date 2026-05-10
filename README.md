# FinSentinel: Financial Risk Intelligence

**DS-UA 301 - Advanced Topics in Data Science · NYU · Spring 2026**

Dori Fu · Rhea Nayar · Fatema Jaynab

---

## Overview
FinSentinel is a financial sentiment classification and market signal pipeline. We wanted to see if prompt-engineered LLMs can extract structured financial risk signals from news headlines, and whether aggregated daily sentiment has any measurable relationship with real stock price returns.

**Research question:** Does few-shot prompting improve over zero-shot for financial sentiment classification? And does aggregated headline sentiment predict next-day stock returns?

**Key result:** FinBERT+LoRA hits **0.978 macro F1** on Financial PhraseBank at **50ms latency** and **zero API cost**, matching GPT-4o-mini few-shot while being 70x smaller and free to run.

## Results Summary
| Model | Accuracy | Macro F1 | Latency | API Cost |
|---|---|---|---|---|
| Majority baseline | 64.5% | 0.340 | -- | Free |
| TF-IDF + LogReg | 90.3% | 0.868 | <1ms | Free |
| FinBERT zero-shot | 96.0% | 0.963 | ~80ms | Free |
| GPT-4o-mini zero-shot | -- | 0.828 | ~1.2s | Paid |
| GPT-4o-mini few-shot | 97.5% | 0.978 | ~1.2s | Paid |
| GPT-4o-mini CoT | 93.0% | 0.941 | ~1.5s | Paid |
| RAG + CoT | -- | 0.953 | ~2.0s | Paid |
| ReAct Agent | -- | 0.260 | ~5s+ | Paid |
| **FinBERT+LoRA** | **98.5%** | **0.978** | **~50ms** | **Free** |
| DistilBERT student | 78.4% | 0.539 | ~10ms | Free |

> Note: All models drop hard on FiQA (accuracy ~10%).

## Pipeline Overview

| Tier | Model | Macro F1 |
|---|---|---|
| 0 | Majority baseline | 0.340 |
| 1 | TF-IDF + Logistic Regression | 0.868 |
| 2A | FinBERT zero-shot | 0.963 |
| 2B | FinBERT + LoRA | **0.978** |
| 3 | GPT-4o-mini (zero/few/CoT/RAG+CoT) | 0.828-0.978 |
| 4 | LangChain ReAct Agent | 0.260 |
| 5 | DistilBERT student (distillation) | 0.539 |


## Dataset

We used the **Financial PhraseBank (AllAgree subset)** - 2,264 financial sentences from OMX Helsinki news, annotated by 16 finance professionals from a retail investor's perspective. Labels are positive, neutral, and negative. The dataset has a class imbalance (61.4% neutral) so we used macro F1 as our main metric.

For cross-domain testing we used **FiQA-Sentiment**, which is Twitter/Stocktwits financial microblogs. Results were not great (see above) but that's an honest finding.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/rn2357-cloud/financial-sentiment-analysis
cd financial-sentiment-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```
### 3. Set your OpenAI API key

```bash
export OPENAI_API_KEY="your-key-here"
```

Or if you're deploying on Streamlit Cloud, add it to your secrets:
```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "your-key-here"
```

The app works without an API key using keyword-based fallback. FinBERT+LoRA inference is completely free.

---

## Running the Notebook

Open `notebooks/FinSentinel.ipynb` in Google Colab or Jupyter. Put `Sentences_AllAgree.txt` in the working directory before running. The notebook goes through all 6 tiers in order.

---

## Running the App

```bash
streamlit run finsentinel_v3.py
```

You can input headlines four ways:
- Live Yahoo Finance headlines by ticker (AAPL, MSFT, NVDA, etc.)
- Paste your own headlines
- Load FiQA directly from HuggingFace
- Upload Sentences_AllAgree.txt for evaluation with ground truth labels

The app classifies each headline and returns sentiment, risk type, severity, market impact, time horizon, confidence, and reasoning. It also aggregates daily sentiment scores per ticker and correlates them against actual stock returns via yfinance.

---

## Key Findings

**Domain beats scale.** FinBERT zero-shot (0.963) beats GPT-4o-mini zero-shot (0.828) even though its 70x smaller. Financial domain pretraining matters a lot more than model size here.

**Few-shot is the biggest lever.** 6 labeled examples boosted macro F1 from 0.828 to 0.978 (p<0.001, McNemar). The examples teach the model to think from a retail investor perspective instead of just picking up on surface words like "sales" or "revenue."

**CoT backfired a little.** CoT (0.941) actually did worse than few-shot (0.978). It made the model too conservative and biased toward neutral predictions. The difference was not statistically significant (p=0.109).

**FinBERT+LoRA is the practical winner.** Same F1 as GPT-4o-mini few-shot, zero API cost, 50ms latency, and only 0.27% of parameters trained. Easy choice for deployment.

**Real-world gap is real.** FPB accuracy (98.5%) collapses to about 10% on FiQA. Clean benchmark performance does not mean the model works on messy real-world text.

---

## Stats

| Comparison | Test | p-value |
|---|---|---|
| Zero-shot vs few-shot | McNemar (n=200) | p < 0.001 |
| Few-shot vs CoT | McNemar (n=200) | p = 0.109 |

Of 29 disagreements between zero-shot and few-shot, few-shot fixed 27 of them and only introduced 2 new errors.

---

## Future Work

- Domain adaptation on noisy text like FiQA and Twitter/Stocktwits
- Bigger distillation dataset 
- Live RSS sentiment-price dashboard
- Better class balancing for negative sentiment

---

## References

- Yang et al. (2020). FinBERT: A Pretrained Language Model for Financial Communications. arXiv:2006.08097
- Deng et al. (2023). What do LLMs Know about Financial Markets? ACM Web Conference.
- Vijayan (2023). A Prompt Engineering Approach for Structured Data Extraction from Unstructured Text Using Conversational LLMs. ACAI 2023.
- Malo et al. (2014). Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts. JASIST.


