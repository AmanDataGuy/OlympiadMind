"""
raptor_builder.py
-----------------
Phase 5.1 — RAPTOR Tree Construction

Builds a hierarchical tree over all 1096 problem texts using:
  1. Sentence-transformers for embeddings
  2. UMAP for dimensionality reduction
  3. GMM for clustering
  4. Mistral 7B on Modal for cluster summarization
  5. FAISS index over all levels combined

Tree structure:
  Level 0 (leaves): individual problem texts
  Level 1: ~100 clusters summarized
  Level 2: ~15 meta-clusters summarized
  Level 3: ~3-5 top-level topic summaries

Output:
  data/raptor/tree.json      — full tree with all nodes and metadata
  data/raptor/raptor.faiss   — FAISS index over all tree levels
  data/raptor/nodes.jsonl    — all nodes in index order
"""

import json
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm


# ── config ────────────────────────────────────────────────────────────────────

DOCUMENTS_PATH    = Path("data/extracted/documents.jsonl")
OUTPUT_DIR        = Path("data/raptor")
TREE_PATH         = OUTPUT_DIR / "tree.json"
FAISS_PATH        = OUTPUT_DIR / "raptor.faiss"
NODES_PATH        = OUTPUT_DIR / "nodes.jsonl"

EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
EMBEDDING_DIM     = 384

# RAPTOR clustering params
UMAP_N_COMPONENTS = 10   # reduce to 10 dims before clustering
MAX_CLUSTER_SIZE  = 50   # max problems per cluster
MIN_CLUSTER_SIZE  = 5    # min problems per cluster

# number of tree levels to build
N_LEVELS          = 3


# ── loaders ───────────────────────────────────────────────────────────────────

def load_problems() -> list:
    """Load all problem texts from documents.jsonl."""
    problems = []
    with open(DOCUMENTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc    = json.loads(line)

            # RAPTOR only over olympiad problems — skip hindi, tabular, multilingual
            if doc.get("source") not in ("olympiad", "cbse", "science_bowl"):
                continue

            pages  = doc.get("pages", [])
            text   = pages[0]["text"] if pages else ""
            if text.strip():
                problems.append({
                    "doc_id":   doc["doc_id"],
                    "source":   doc.get("source", ""),
                    "doc_type": doc.get("doc_type", ""),
                    "year":     doc.get("year", ""),
                    "competition": doc.get("competition", ""),
                    "text":     text[:500],
                    "level":    0,
                })
    return problems


# ── embedding ─────────────────────────────────────────────────────────────────

def embed_texts(texts: list, model) -> np.ndarray:
    """Embed a list of texts using sentence-transformers."""
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype(np.float32)


# ── clustering ────────────────────────────────────────────────────────────────

def cluster_embeddings(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    UMAP dimensionality reduction followed by GMM clustering.
    Returns cluster labels for each embedding.
    """
    import umap
    from sklearn.mixture import GaussianMixture

    print(f"  UMAP: {embeddings.shape[1]}d → {UMAP_N_COMPONENTS}d...")
    reducer = umap.UMAP(
        n_components  = UMAP_N_COMPONENTS,
        n_neighbors   = 15,
        min_dist      = 0.1,
        random_state  = 42,
    )
    reduced = reducer.fit_transform(embeddings)

    print(f"  GMM clustering into {n_clusters} clusters...")
    gmm = GaussianMixture(
        n_components  = n_clusters,
        covariance_type = "full",
        random_state  = 42,
        max_iter      = 100,
    )
    gmm.fit(reduced)
    labels = gmm.predict(reduced)
    return labels


def estimate_n_clusters(n_items: int) -> int:
    """Estimate number of clusters — aim for ~MAX_CLUSTER_SIZE items per cluster."""
    n = max(2, n_items // MAX_CLUSTER_SIZE)
    return min(n, n_items // MIN_CLUSTER_SIZE)


# ── summarization ─────────────────────────────────────────────────────────────

def summarize_cluster_local(texts: list) -> str:
    """
    Summarize a cluster of problem texts using a simple extractive approach.
    Used as fallback if Modal is not available.
    """
    # take first sentence from each text, join
    snippets = []
    for t in texts[:5]:
        first_sentence = t.split(".")[0].strip()
        if first_sentence:
            snippets.append(first_sentence)
    return ". ".join(snippets) + "."


def summarize_clusters_modal(cluster_groups: list) -> list:
    """
    Summarize all clusters using Mistral 7B on Modal.
    Each cluster_group is a list of problem texts.
    Returns list of summary strings in same order.
    """
    import modal

    @modal.function(
        image=modal.Image.debian_slim().pip_install("transformers", "torch", "accelerate"),
        gpu="A10G",
        timeout=1800,
    )
    def summarize_batch(groups: list) -> list:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_id  = "mistralai/Mistral-7B-Instruct-v0.3"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model     = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
        model.eval()

        summaries = []
        for texts in groups:
            combined = "\n".join(f"- {t[:200]}" for t in texts[:10])
            prompt   = f"[INST] Summarize these physics problems into one paragraph describing the common physics concepts and topics: \n{combined} [/INST]"
            inputs   = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=150, do_sample=False)
            generated = output[0][inputs["input_ids"].shape[1]:]
            summary   = tokenizer.decode(generated, skip_special_tokens=True).strip()
            summaries.append(summary)

        return summaries

    with modal.enable_output():
        summaries = list(summarize_batch.remote(cluster_groups))

    return summaries


# ── raptor tree builder ───────────────────────────────────────────────────────

def build_raptor_level(nodes: list, model, level: int, use_modal: bool = True) -> list:
    """
    Build one level of the RAPTOR tree.

    Takes a list of nodes from the previous level, clusters them,
    summarizes each cluster, and returns the new parent nodes.
    """
    print(f"\nBuilding Level {level} ({len(nodes)} nodes)...")

    texts      = [n["text"] for n in nodes]
    embeddings = embed_texts(texts, model)

    n_clusters = estimate_n_clusters(len(nodes))
    print(f"  Clustering into {n_clusters} groups...")
    labels     = cluster_embeddings(embeddings, n_clusters)

    # group nodes by cluster
    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(nodes[i])

    print(f"  Summarizing {len(clusters)} clusters...")

    # prepare texts for summarization
    cluster_texts = [[n["text"] for n in cluster_nodes] for cluster_nodes in clusters.values()]

    if use_modal:
        try:
            summaries = summarize_clusters_modal(cluster_texts)
        except Exception as e:
            print(f"  Modal failed ({e}), using local fallback...")
            summaries = [summarize_cluster_local(texts) for texts in cluster_texts]
    else:
        summaries = [summarize_cluster_local(texts) for texts in cluster_texts]

    # create parent nodes
    parent_nodes = []
    for cluster_id, (cluster_nodes, summary) in enumerate(zip(clusters.values(), summaries)):
        child_ids = [n["node_id"] for n in cluster_nodes]
        parent_nodes.append({
            "node_id":    f"L{level}_C{cluster_id}",
            "level":      level,
            "text":       summary,
            "children":   child_ids,
            "cluster_id": cluster_id,
            "n_children": len(child_ids),
        })

    return parent_nodes, clusters


def build_raptor_tree(problems: list, model, use_modal: bool = True) -> dict:
    """
    Build the full RAPTOR tree bottom-up.
    Returns dict with all_nodes (flat list) and tree structure.
    """
    # assign node IDs to leaf nodes
    leaf_nodes = []
    for i, p in enumerate(problems):
        leaf_nodes.append({
            "node_id":     f"L0_{p['doc_id']}",
            "level":       0,
            "text":        p["text"],
            "doc_id":      p["doc_id"],
            "source":      p["source"],
            "competition": p["competition"],
            "year":        p["year"],
            "children":    [],
        })

    all_nodes    = leaf_nodes.copy()
    current_nodes = leaf_nodes
    tree_levels  = {0: leaf_nodes}

    for level in range(1, N_LEVELS + 1):
        if len(current_nodes) < MIN_CLUSTER_SIZE * 2:
            print(f"\nOnly {len(current_nodes)} nodes — stopping tree at level {level-1}")
            break

        parent_nodes, clusters = build_raptor_level(current_nodes, model, level, use_modal)
        all_nodes.extend(parent_nodes)
        tree_levels[level] = parent_nodes
        current_nodes      = parent_nodes

    return {
        "levels":    tree_levels,
        "all_nodes": all_nodes,
        "n_levels":  len(tree_levels),
        "n_nodes":   len(all_nodes),
    }


# ── faiss index ───────────────────────────────────────────────────────────────

def build_raptor_index(all_nodes: list, model) -> faiss.Index:
    """Build FAISS index over all tree nodes (all levels combined)."""
    texts      = [n["text"] for n in all_nodes]
    embeddings = embed_texts(texts, model)

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    return index


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading problems...")
    problems = load_problems()
    print(f"  {len(problems)} problems loaded\n")

    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded.\n")

    print("Building RAPTOR tree...")
    tree = build_raptor_tree(problems, model, use_modal=True)

    print(f"\nTree built:")
    for level, nodes in tree["levels"].items():
        print(f"  Level {level}: {len(nodes)} nodes")

    # save tree
    with open(TREE_PATH, "w", encoding="utf-8") as f:
        # save without embeddings — just metadata and text
        json.dump({
            "n_levels": tree["n_levels"],
            "n_nodes":  tree["n_nodes"],
            "levels": {
                str(k): v for k, v in tree["levels"].items()
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"\nTree saved to: {TREE_PATH}")

    # build and save FAISS index
    print("\nBuilding FAISS index over all tree nodes...")
    index = build_raptor_index(tree["all_nodes"], model)
    faiss.write_index(index, str(FAISS_PATH))
    print(f"FAISS index saved to: {FAISS_PATH}")

    # save nodes in index order
    with open(NODES_PATH, "w", encoding="utf-8") as f:
        for node in tree["all_nodes"]:
            f.write(json.dumps(node, ensure_ascii=False) + "\n")
    print(f"Nodes saved to: {NODES_PATH}")

    print(f"\nDone.")
    print(f"  Total nodes indexed : {tree['n_nodes']}")
    print(f"  Tree levels         : {tree['n_levels']}")


if __name__ == "__main__":
    main()