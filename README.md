# Ta Cabrones Menu Bot

A small **RAG** (Retrieval-Augmented Generation) chatbot that answers customer questions about the menu of *Ta Cabrones*, a Mexican restaurant, plus a set of **evals** (tests that measure how good the bot's answers are).

The menu is in Spanish, so most questions and answers are in Spanish too.

> **How RAG works, in one analogy:** an open-book exam. The menu PDF is the book, the *chunks* are its pages, the *embeddings* are a topic index, the vector store is the shelf, and the LLM is the student who looks up the right pages before answering.

## Project layout

| File / folder | What it is |
|---|---|
| `rag.py` | The bot. Builds (or loads) the search index from `data/` and answers questions in a chat loop. |
| `data/Menu_Ta_Cabrones_Texto.pdf` | The restaurant menu the bot reads from. |
| `storage/` | The saved index, so the menu isn't re-embedded on every run. |
| `test_rag.py` | pytest suite: an LLM judge grades the answers, and an exact-match check grades the prices. |
| `goldens.json` | Normal menu questions with their correct answers. |
| `edge_cases.json` | Tricky questions: typos, English, items with several prices. |
| `eval_metrics.py` | Asks "is X on the menu?" for each item in `menu_items.json` and builds a confusion matrix. |
| `menu_items.json` | Items labelled as on the menu or not on the menu. |
| `results.csv` | Output of `eval_metrics.py`, one row per item. |
| `judge_check.py` | Tests the **judge**, not the bot: feeds it answers we already know are right or wrong. |
| `retrieval_eval.py` | Compares chunking strategies with Recall@K, MRR and NDCG. |

## Setup

Requires Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root (it is git-ignored, never commit it):

```
OPENAI_API_KEY=your-key-here
TYPESAFE_API_KEY=your-key-here
```

- `OPENAI_API_KEY` is used by the bot's LLM and embedding model.
- `TYPESAFE_API_KEY` is used by the Jev judge in `test_rag.py`.

## Run the bot

```bash
python rag.py
```

Type a question (e.g. `¿Cuánto cuesta la hamburguesa en combo?`) and type `salir` to quit.

The first run builds the index and saves it to `storage/`. **If you change the PDF in `data/` or the embedding model, delete `storage/`** so the index is rebuilt.

## Run the evals

### 1. Answer quality: `pytest`

```bash
pytest test_rag.py -v
```

There are three tests:

- **`test_menu_question`** and **`test_edge_case`** use an *LLM-as-judge* (an AI model that grades another AI's answers). The judge asks four yes/no questions about every answer:
  1. Does it directly answer the question?
  2. Is every fact in it found in the retrieved menu text? (no made-up facts)
  3. Does it match the expected answer, especially on prices?
  4. Is it polite?

  With `strict_mode=True`, every question has to pass.
- **`test_exact_match`** is a plain rule-based check, with no AI involved: every price the bot says has to appear in the expected answer.

### 2. Yes/no accuracy: confusion matrix

```bash
python eval_metrics.py
```

This asks the bot whether each item is on the menu, then prints:

- **TP / FP / FN / TN**, the confusion matrix. An FP means the bot invented a dish, and an FN means it lost a sale.
- **Accuracy, precision, recall and F1.**

It saves the per-item results to `results.csv`.

### 3. Checking the judge: meta-eval

```bash
python judge_check.py
```

A judge is only useful if it is right. This script feeds it seven hand-written answers (correct, wrong price, made-up extras, rude, and so on) and reports how many of its verdicts agree with ours.

### 4. Retrieval quality

```bash
python retrieval_eval.py
```

This checks whether the *search step* finds the right menu text before the LLM sees it. It compares three chunking strategies:

- 100-token chunks with no overlap
- 100-token chunks with 30 tokens of overlap
- one chunk per dish

It scores each strategy with:

- **Recall@3**: did the needed text appear anywhere in the top 3 results?
- **MRR**: how high did the first correct result rank?
- **NDCG@3**: how well were *all* the correct results ranked?

## Tech stack

- [LlamaIndex](https://www.llamaindex.ai/): loads the documents, builds the index and handles retrieval
- OpenAI embeddings (`text-embedding-3-small`)
- [DeepEval](https://github.com/confident-ai/deepeval): test cases, datasets and the Jev judge metric
- pytest
