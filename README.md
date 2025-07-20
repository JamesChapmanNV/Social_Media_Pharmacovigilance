# Pharmacovigilance on Social Media  
_LLMs + Retrieval‑Augmented Generation for Identifying Illicit Drug Use_

> **CS 830: Advanced Topics in AI – Term Project**  
> James Chapman · Kansas State University  

---

## Project Overview  
  Detecting and **disambiguating drug references on social‑media posts** is hard: slang evolves quickly, many terms are ambiguous (“Snow”, “Dabs”, “Fettuccine”), and context determines whether use is illicit, or simply a metaphor/news/discouraging use.
This repo explores a **Contextualized Retrieval‑Augmented Generation (RAG) pipeline** that:

1. Harvests synonym and contextual knowledge for 77 substances of interest.  
2. Builds multiple retrievers (FAISS, BM25, hybrid + Cohere reranker).  
3. Runs lightweight & local LLMs (GPT‑4o‑mini, o4‑mini, Llama‑3 8B, Qwen‑4B) to  
   * binary‑classify posts (illicit/abusive **T** vs **F**)  
   * link each detected phrase back to its canonical drug entity.  
