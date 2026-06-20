"""Compute real knowledge-graph statistics over the live catalogue.

Builds the KG exactly as the recommender/agent does, then reports graph size,
edge-type counts, how many products are eligible for and affected by the
conflict/synergy layer, and concrete penalty/boost examples.
"""
from collections import Counter

from agent.tools import _load_product_data
from labeling.knowledge_graph import build_kg, conflict_synergy_factor


def main():
    products, reviews = _load_product_data()
    G = build_kg(products, reviews)

    rel_counts = Counter(d.get("rel") for _, _, d in G.edges(data=True))
    kind_counts = Counter(d.get("kind") for _, d in G.nodes(data=True))

    print(f"Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")
    print("Node kinds:", dict(kind_counts))
    print("Edge rels :", dict(rel_counts))

    n_products = sum(1 for _, d in G.nodes(data=True) if d.get("kind") == "product")
    eligible = boosted = penalized = neutral = 0
    factors = []
    conflict_examples, synergy_examples = [], []

    for n, d in G.nodes(data=True):
        if d.get("kind") != "product":
            continue
        url = n[1]
        actives = {m[1] for _, m, e in G.out_edges(n, data=True) if e.get("rel") == "HAS_ACTIVE"}
        if len(actives) >= 2:
            eligible += 1
        factor, reasons = conflict_synergy_factor(G, url)
        factors.append(factor)
        if factor > 1.0:
            boosted += 1
            if len(synergy_examples) < 3 and any("synergy" in r for r in reasons):
                synergy_examples.append((d.get("label"), factor, reasons))
        elif factor < 1.0:
            penalized += 1
            if len(conflict_examples) < 3 and any("conflict" in r for r in reasons):
                conflict_examples.append((d.get("label"), factor, reasons))
        else:
            neutral += 1

    print(f"\nProducts: {n_products}")
    print(f"Eligible (>=2 actives): {eligible}")
    print(f"Boosted (factor>1): {boosted}")
    print(f"Penalized (factor<1): {penalized}")
    print(f"Neutral (factor=1): {neutral}")
    nz = [f for f in factors if f != 1.0]
    if nz:
        print(f"Factor range (non-neutral): {min(nz):.4f} .. {max(nz):.4f}")

    print("\n--- Example conflicts ---")
    for label, factor, reasons in conflict_examples:
        print(f"[{factor}] {label}")
        for r in reasons:
            print("   ", r)
    print("\n--- Example synergies ---")
    for label, factor, reasons in synergy_examples:
        print(f"[{factor}] {label}")
        for r in reasons:
            print("   ", r)


if __name__ == "__main__":
    main()
