# FinSentinel: Financial Risk Intelligence

**DS-UA 301 - Advanced Topics in Data Science · NYU · Spring 2026**

Dori Fu · Rhea Nayar · Fatema Jaynab

---

## Overview
FinSentinel is a financial sentiment classification and market signal pipeline. We wanted to see if prompt-engineered LLMs can extract structured financial risk signals from news headlines, and whether aggregated daily sentiment has any measurable relationship with real stock price returns.

**Research question:** Does few-shot prompting improve over zero-shot for financial sentiment classification? And does aggregated headline sentiment predict next-day stock returns?

**Key result:** FinBERT+LoRA hits **0.978 macro F1** on Financial PhraseBank at **50ms latency** and **zero API cost**, matching GPT-4o-mini few-shot while being 70x smaller and free to run.

