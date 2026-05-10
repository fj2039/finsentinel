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

> Note: All models drop hard on FiQA (accuracy ~10%). Good FPB results do not transfer to informal microblog financial text.
