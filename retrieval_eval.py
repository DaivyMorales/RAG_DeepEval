import math

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

import rag  # noqa: F401  (sets up the same embedding model as the bot)

K = 3

# For each question: text snippets that must appear in the retrieved chunks.
DATASET = [
    ("¿Cuánto cuesta la hamburguesa en combo?", ["Hamburguesa Carne", "Combo $35.000"]),
    ("¿Cuánto cuesta el Birramen?", ["Birramen Caldo"]),
    ("¿Qué lleva el taco al pastor?", ["Al pastor Carne de cerdo"]),
    ("¿Cuánto vale la horchata?", ["Horchata"]),
    ("¿Cuánto cuestan los Dorilocos familiares?", ["Familiar"]),
    ("¿Qué sabores de jugo tienen?", ["Sabores de jugo"]),
    ("¿Cuánto cuesta el burrito sin costra?", ["Burritos Tortillas", "Sin costra $29.000"]),
    ("¿Tienen cerveza?", ["Coronita", "Cola y Pola"]),
]


def clean(text):
    return " ".join(text.split())  # ignore line breaks and double spaces


def is_relevant(chunk, snippets):
    return any(s in clean(chunk) for s in snippets)


def recall_at_k(retrieved, snippets):
    """What fraction of the needed snippets showed up anywhere in the top K?"""
    found = [s for s in snippets if any(s in clean(c) for c in retrieved)]
    return len(found) / len(snippets)


def reciprocal_rank(retrieved, snippets):
    """1 if the first relevant chunk is 1st, 1/2 if 2nd, 1/3 if 3rd, 0 if missing."""
    for rank, chunk in enumerate(retrieved, start=1):
        if is_relevant(chunk, snippets):
            return 1 / rank
    return 0


def ndcg(retrieved, snippets, all_chunks):
    """Like MRR, but rewards EVERY relevant chunk, with less credit the lower it ranks."""
    dcg = sum(1 / math.log2(rank + 1)
              for rank, chunk in enumerate(retrieved, start=1) if is_relevant(chunk, snippets))
    n_relevant = min(K, sum(is_relevant(c, snippets) for c in all_chunks))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, n_relevant + 1))
    return dcg / ideal if ideal else 0


def chunk_by_dish(documents):
    """One chunk per dish, with its section name on top (e.g. "DORILOCOS Familiar $45.000")."""
    chunks, section, current, prev = [], "", [], ""

    def close():
        if current:
            chunks.append(TextNode(text=section + "\n" + "\n".join(current)))
            current.clear()

    for doc in documents:
        for line in doc.text.split("\n"):
            line = line.strip()
            # Skip page headers like "Ta Cabrones · Menú informativo" and page numbers.
            if not line or line.isdigit() or "ta cabrones" in line.lower() or line.startswith("Menú ·"):
                continue
            if line.isupper():  # section title: PLATOS, BEBIDAS, TACOS...
                close()
                section = line
            # A new dish starts on the line after a price, a "Combo: ..." line or a note ending in ".".
            elif ("$" in prev or prev.startswith("Combo:") or prev.endswith(".")) \
                    and "$" not in line and not line.startswith("Combo:"):
                close()
                current.append(line)
            else:
                current.append(line)
            prev = line
    close()
    return chunks


def evaluate(nodes, show_details=False):
    all_chunks = [n.get_content() for n in nodes]
    retriever = VectorStoreIndex(nodes).as_retriever(similarity_top_k=K)

    totals = {"recall": 0, "rr": 0, "ndcg": 0}
    for question, snippets in DATASET:
        retrieved = [n.get_content() for n in retriever.retrieve(question)]
        r, rr, n = recall_at_k(retrieved, snippets), reciprocal_rank(retrieved, snippets), ndcg(retrieved, snippets, all_chunks)
        totals["recall"] += r
        totals["rr"] += rr
        totals["ndcg"] += n

        if show_details:
            print(f"\n{question}   Recall@{K}={r:.2f}  RR={rr:.2f}  NDCG={n:.2f}")
            for rank, chunk in enumerate(retrieved, start=1):
                mark = "✅" if is_relevant(chunk, snippets) else "  "
                print(f"   {rank}. {mark} {clean(chunk)[:80]}")

    return {name: total / len(DATASET) for name, total in totals.items()}, len(nodes)


if __name__ == "__main__":
    documents = SimpleDirectoryReader("data").load_data()
    strategies = {
        "100 tokens, overlap 0": SentenceSplitter(chunk_size=100, chunk_overlap=0).get_nodes_from_documents(documents),
        "100 tokens, overlap 30": SentenceSplitter(chunk_size=100, chunk_overlap=30).get_nodes_from_documents(documents),
        "one chunk per dish": chunk_by_dish(documents),
    }

    results = {}
    for name, nodes in strategies.items():
        results[name] = evaluate(nodes, show_details=(name == "one chunk per dish"))

    print(f"\n{'Strategy':<24} {'chunks':>6} {'Recall@' + str(K):>9} {'MRR':>6} {'NDCG@' + str(K):>7}")
    for name, (m, n_chunks) in results.items():
        print(f"{name:<24} {n_chunks:>6} {m['recall']:>9.2f} {m['rr']:>6.2f} {m['ndcg']:>7.2f}")
