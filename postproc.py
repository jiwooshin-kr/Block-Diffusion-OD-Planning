"""
Path post-processing operators, shared by every evaluation script.

Post-processing has two distinct axes and conflating them leads to wrong claims:

  REPAIR          fixes a failure, so it CHANGES the feasibility metrics and only
                  ever ADDS nodes. Reporting valid&arrival after repair is
                  meaningless: it saturates to 1.000 for every arm.
    splice()      P1, validity repair. Replaces an illegal jump u->v by the legal
                  walk u->...->v. No node of the model output is deleted.
    endpoint()    P3, arrival repair. Appends the shortest path from the last
                  vertex to the destination. Touches only the tail.

  SIMPLIFICATION  fixes nothing, so arrival is preserved exactly and validity is
                  non-decreasing; it only REMOVES nodes. Safe to apply
                  unconditionally, and it does not launder the headline metrics.
    simplify_path()  loop cut. Re-exported from models_seq.bd_models, where the
                  samplers apply it to their returned paths by default.

Before this module the three active eval scripts each carried a private copy of
splice/endpoint (logically identical, differing only in a variable name). They now
import from here. The copies under legacy/v2_round/ are deliberately left alone:
CLAUDE.md keeps that directory as the record of superseded rounds.

All operators take the SCENARIO graph/adjacency (except_0 in the v6 protocol), not
the training graph, so repairs are legal in the scenario being evaluated.
"""

import networkx as nx

from models_seq.bd_models import simplify_path  # noqa: F401  (re-exported)

__all__ = ["splice", "endpoint", "simplify_path"]


def splice(p, A, G):
    """P1, validity repair: replace every illegal edge by a legal walk.

    Walks the consecutive pairs (u, v) of p. A legal pair is kept as is; an illegal
    one is replaced by nx.shortest_path(G, u, v), which inserts the intermediate
    vertices. Nothing is deleted, so every vertex the model emitted survives and the
    length only grows.

    Returns (repaired_path, n_added) where n_added counts inserted intermediate
    vertices (the shortest-path segment includes v, which was already present).

    Fallback: if u and v are disconnected in G the illegal edge is left in place.
    Measured on v6 this never fired -- the P1 tables report valid = 1.000 exactly.
    """
    out, added = [p[0]], 0
    for u, v in zip(p[:-1], p[1:]):
        if A[u, v]:
            out.append(v)
            continue
        try:
            seg = nx.shortest_path(G, u, v)[1:]
            added += len(seg) - 1
            out.extend(seg)
        except Exception:
            out.append(v)
    return out, added


def endpoint(p, dst, G):
    """P3, arrival repair: extend the tail to the destination.

    A path that already ends at dst (or is empty) is returned untouched; otherwise
    the shortest path from p[-1] to dst is appended. The interior is never modified.

    Returns (repaired_path, n_added). Here n_added is the full segment length, not
    length-1 as in splice(), because dst was not part of the original path.
    """
    if len(p) == 0 or p[-1] == dst:
        return list(p), 0
    try:
        seg = nx.shortest_path(G, p[-1], dst)[1:]
        return list(p) + seg, len(seg)
    except Exception:
        return list(p), 0
