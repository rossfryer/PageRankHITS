import json
import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
from scipy import sparse
import plotly.graph_objects as go

# -----------------------------
# Utilities
# -----------------------------

def l1_norm(x):
    return float(np.sum(np.abs(x)))


def normalize(vec, p="l1"):
    if p == "l2":
        n = np.linalg.norm(vec)
        return vec / (n if n != 0 else 1.0)
    # default L1
    s = np.sum(np.abs(vec))
    return vec / (s if s != 0 else 1.0)


def to_transition_matrix(G, node_index):
    """Return row-stochastic transition matrix for PageRank (n x n),
    handling dangling rows by leaving 0s (handled in power iteration loop)."""
    n = len(node_index)
    rows, cols, data = [], [], []
    for u in G.nodes():
        i = node_index[u]
        out_neighbors = list(G.successors(u))
        if len(out_neighbors) > 0:
            w = 1.0 / len(out_neighbors)
            for v in out_neighbors:
                j = node_index[v]
                rows.append(i)
                cols.append(j)
                data.append(w)
    if len(data) == 0:
        # empty/edgeless graph
        return sparse.csr_matrix((n, n), dtype=float)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)


def pagerank_power_iteration(G, d=0.85, tol=1e-6, max_iter=100, personalization=None):
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {"iterations": [], "final": {}, "converged": True}

    node_index = {node: i for i, node in enumerate(nodes)}

    P = to_transition_matrix(G, node_index)  # row-stochastic where defined
    v = np.ones(n) / n  # uniform start

    if personalization is None:
        p = np.ones(n) / n
    else:
        p = np.array([personalization.get(node, 0.0) for node in nodes], dtype=float)
        p = p / (np.sum(p) if np.sum(p) > 0 else 1.0)

    iters = []
    converged = False

    for k in range(max_iter):
        # Handle dangling nodes: rows with sum 0 get replaced by uniform distribution
        row_sums = np.array(P.sum(axis=1)).flatten()
        dangling = (row_sums == 0)

        # Compute next vector: v_next = d*(v*P + v*D*(1/n)) + (1-d)*p
        # Convert sparse matrix to dense for easier operations
        P_dense = P.toarray()
        base = v.dot(P_dense)
        if np.any(dangling):
            mass = np.sum(v[dangling])
            base = base + mass * (np.ones(n) / n)
        v_next = d * base + (1 - d) * p

        delta = l1_norm(v_next - v)

        iters.append({
            "k": k + 1,
            "vector": v_next.copy(),
            "delta_l1": float(delta),
            "dangling_mass": float(np.sum(v[dangling])) if np.any(dangling) else 0.0,
        })

        v = v_next
        if delta < tol:
            converged = True
            break

    result = {nodes[i]: float(v[i]) for i in range(n)}
    return {"iterations": iters, "final": result, "converged": converged, "transition": P, "nodes": nodes}


def hits_power_iteration(G, tol=1e-6, max_iter=100, norm="l1"):
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {"iterations": [], "final_h": {}, "final_a": {}, "converged": True}

    node_index = {node: i for i, node in enumerate(nodes)}

    # Build adjacency matrix A where A[i,j] = 1 if i -> j
    rows, cols, data = [], [], []
    for u, v in G.edges():
        rows.append(node_index[u])
        cols.append(node_index[v])
        data.append(1.0)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)

    h = np.ones(n) / n
    a = np.ones(n) / n

    iters = []
    converged = False

    for k in range(max_iter):
        a_new = A.T.dot(h)
        h_new = A.dot(a_new)
        a_new = normalize(a_new, p=norm)
        h_new = normalize(h_new, p=norm)
        delta = l1_norm(a_new - a) + l1_norm(h_new - h)

        iters.append({
            "k": k + 1,
            "a": a_new.copy(),
            "h": h_new.copy(),
            "delta_l1": float(delta)
        })

        a, h = a_new, h_new
        if delta < tol:
            converged = True
            break

    final_a = {nodes[i]: float(a[i]) for i in range(n)}
    final_h = {nodes[i]: float(h[i]) for i in range(n)}
    return {"iterations": iters, "final_a": final_a, "final_h": final_h, "converged": converged, "A": A, "nodes": nodes}


# -----------------------------
# Visualization helpers
# -----------------------------

def plot_graph(G, scores=None, title="Graph"):
    if len(G.nodes()) == 0:
        return go.Figure()

    pos = nx.spring_layout(G, seed=42)

    node_x, node_y, node_text, node_size = [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        score = scores.get(n, 0.0) if scores else 0.0
        node_text.append(f"{n}<br>score={score:.4f}\n(in={G.in_degree(n)}, out={G.out_degree(n)})")
        node_size.append(20 + 60 * (score if scores else 0.0))

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1),
        hoverinfo='none',
        mode='lines')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[str(n) for n in G.nodes()],
        textposition='bottom center',
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(size=node_size, line=dict(width=1)))

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title=title, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
    return fig


# -----------------------------
# App state & presets
# -----------------------------

def init_state():
    st.session_state.setdefault("graph", nx.DiGraph())
    st.session_state.setdefault("algo", "PageRank")
    st.session_state.setdefault("params", {
        "d": 0.85,
        "tol": 1e-6,
        "max_iter": 100,
        "hits_norm": "l1"
    })
    st.session_state.setdefault("iterations", [])
    st.session_state.setdefault("iter_index", 0)
    st.session_state.setdefault("result", {})
    st.session_state.setdefault("activity", [])


def log(msg):
    st.session_state["activity"].append({
        "time": datetime.now().isoformat(timespec='seconds'),
        "msg": msg
    })


def set_preset(name):
    G = nx.DiGraph()
    if name == "star":
        G.add_nodes_from(["A", "B", "C", "D"])  # A points to all
        G.add_edges_from([("A", "B"), ("A", "C"), ("A", "D")])
    elif name == "chain":
        G.add_nodes_from(["A", "B", "C", "D"])  # A->B->C->D
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    elif name == "two_clusters":
        G.add_nodes_from(["A1", "A2", "B1", "B2"])  # clusters A and B with a bridge
        G.add_edges_from([("A1", "A2"), ("A2", "A1"), ("B1", "B2"), ("B2", "B1"), ("A2", "B1")])
    elif name == "dangling":
        G.add_nodes_from(["A", "B", "C"])  # C is dangling
        G.add_edges_from([("A", "B"), ("B", "A")])
    elif name == "incoming_range":
        # Shows range of incoming links: 0, 1, 2, 3, 4 incoming links with 5 nodes
        G.add_nodes_from(["A", "B", "C", "D", "E"])
        # A has 0 incoming links (dangling)
        # B has 1 incoming link
        G.add_edges_from([("A", "B")])
        # C has 2 incoming links  
        G.add_edges_from([("A", "C"), ("B", "C")])
        # D has 3 incoming links
        G.add_edges_from([("A", "D"), ("B", "D"), ("C", "D")])
        # E has 4 incoming links
        G.add_edges_from([("A", "E"), ("B", "E"), ("C", "E"), ("D", "E")])
    else:
        G.add_node("A")
    st.session_state["graph"] = G
    st.session_state["iterations"] = []
    st.session_state["iter_index"] = 0
    st.session_state["result"] = {}
    log(f"Loaded preset: {name}")


# -----------------------------
# Run algorithms & rebuild iterations
# -----------------------------

def run_algorithm():
    G = st.session_state["graph"]
    algo = st.session_state["algo"]
    p = st.session_state["params"]

    try:
        if algo == "PageRank":
            out = pagerank_power_iteration(G, d=p["d"], tol=p["tol"], max_iter=p["max_iter"])
            st.session_state["iterations"] = out["iterations"]
            st.session_state["result"] = out["final"]
            st.session_state["nodes_order"] = out.get("nodes", list(G.nodes()))
        else:
            out = hits_power_iteration(G, tol=p["tol"], max_iter=p["max_iter"], norm=p["hits_norm"])
            st.session_state["iterations"] = out["iterations"]
            st.session_state["result"] = {  # store both for convenience
                "authority": out["final_a"],
                "hub": out["final_h"],
            }
            st.session_state["nodes_order"] = out.get("nodes", list(G.nodes()))

        st.session_state["iter_index"] = 0
        log(f"Ran {algo} with params {p}")
        st.success("Algorithm completed successfully!")
        
    except Exception as e:
        st.error(f"Error running {algo}: {e}")
        log(f"Error running {algo}: {e}")


# -----------------------------
# App UI
# -----------------------------

def main():
    st.set_page_config(page_title="PageRank & HITS Tutor (MVP)", layout="wide")
    init_state()

    st.title("PageRank & HITS Interactive Tutor — MVP")

    # Sidebar controls
    with st.sidebar:
        st.header("Mode & Algorithm")
        algo = st.radio("Algorithm", ["PageRank", "HITS"], index=0, key="algo")

        st.header("Parameters")
        tol = st.number_input("Convergence tolerance (ε)", value=float(st.session_state["params"]["tol"]), min_value=1e-12, max_value=1e-2, step=1e-6, format="%e")
        max_iter = st.number_input("Max iterations", value=int(st.session_state["params"]["max_iter"]), min_value=1, max_value=1000, step=1)
        st.session_state["params"]["tol"] = float(tol)
        st.session_state["params"]["max_iter"] = int(max_iter)

        if algo == "PageRank":
            d = st.slider("Damping factor d", min_value=0.5, max_value=0.99, value=float(st.session_state["params"]["d"]))
            st.session_state["params"]["d"] = float(d)
        else:
            hits_norm = st.radio("HITS normalization", ["l1", "l2"], index=0)
            st.session_state["params"]["hits_norm"] = hits_norm

        st.divider()
        st.header("Presets")
        colp1, colp2 = st.columns(2)
        with colp1:
            if st.button("Star"):
                set_preset("star")
        with colp2:
            if st.button("Chain"):
                set_preset("chain")
        colp3, colp4 = st.columns(2)
        with colp3:
            if st.button("Two clusters"):
                set_preset("two_clusters")
        with colp4:
            if st.button("Dangling"):
                set_preset("dangling")
        colp5, colp6 = st.columns(2)
        with colp5:
            if st.button("Incoming Range"):
                set_preset("incoming_range")
        with colp6:
            st.empty()  # Empty column for symmetry

        st.divider()
        st.header("Import / Export")
        uploaded = st.file_uploader("Load graph JSON", type=["json"])
        if uploaded is not None:
            try:
                data = json.load(uploaded)
                G = nx.DiGraph()
                G.add_nodes_from(data.get("nodes", []))
                G.add_edges_from([tuple(e) for e in data.get("edges", [])])
                st.session_state["graph"] = G
                st.success("Graph loaded")
                log("Loaded graph from JSON upload")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                st.error(f"Failed to load JSON: {e}")
            except Exception as e:
                st.error(f"Unexpected error loading JSON: {e}")

        if st.button("Download current graph JSON"):
            G = st.session_state["graph"]
            payload = {
                "nodes": list(G.nodes()),
                "edges": [[u, v] for u, v in G.edges()],
                "params": st.session_state["params"],
            }
            buf = io.StringIO()
            json.dump(payload, buf, indent=2)
            st.download_button("Save graph.json", data=buf.getvalue(), file_name="graph.json", mime="application/json")

        if st.button("Run algorithm"):
            run_algorithm()

    # Main layout
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Graph Editor")
        with st.expander("Add / Remove nodes & edges", expanded=True):
            cols = st.columns(3)
            with cols[0]:
                new_node = st.text_input("New node label", value="")
                if st.button("Add node") and new_node:
                    st.session_state["graph"].add_node(new_node)
                    log(f"Added node {new_node}")
            with cols[1]:
                rm_node = st.text_input("Remove node label", value="")
                if st.button("Remove node") and rm_node in st.session_state["graph"].nodes:
                    st.session_state["graph"].remove_node(rm_node)
                    log(f"Removed node {rm_node}")
            with cols[2]:
                if st.button("Reset graph"):
                    st.session_state["graph"] = nx.DiGraph()
                    st.session_state["iterations"] = []
                    st.session_state["result"] = {}
                    st.session_state["iter_index"] = 0
                    log("Reset graph")

            st.markdown("**Add / Remove directed edge**")
            cols2 = st.columns(4)
            with cols2[0]:
                eu = st.text_input("from", key="edge_from")
            with cols2[1]:
                ev = st.text_input("to", key="edge_to")
            with cols2[2]:
                if st.button("Add edge") and eu and ev:
                    if eu in st.session_state["graph"].nodes and ev in st.session_state["graph"].nodes:
                        st.session_state["graph"].add_edge(eu, ev)
                        log(f"Added edge {eu}->{ev}")
                    else:
                        st.warning("Both nodes must exist")
            with cols2[3]:
                if st.button("Remove edge") and eu and ev:
                    if st.session_state["graph"].has_edge(eu, ev):
                        st.session_state["graph"].remove_edge(eu, ev)
                        log(f"Removed edge {eu}->{ev}")

        # Current graph figure
        st.plotly_chart(plot_graph(st.session_state["graph"],
                                   scores=st.session_state.get("result", {}) if st.session_state["algo"] == "PageRank" else st.session_state.get("result", {}).get("authority", {}),
                                   title="Graph (node size by score)"), use_container_width=True)

        # Iteration controls & table
        if st.session_state["iterations"]:
            st.subheader("Iterations")
            maxk = len(st.session_state["iterations"]) - 1
            idx = st.slider("Iteration", 0, maxk, value=st.session_state["iter_index"], key="iter_slider")

            if idx < len(st.session_state["iterations"]):
                it = st.session_state["iterations"][idx]
                nodes = st.session_state.get("nodes_order", list(st.session_state["graph"].nodes()))
            else:
                st.warning("Invalid iteration index")
                return

            if st.session_state["algo"] == "PageRank":
                if "vector" in it:
                    vec = it["vector"]
                    df = pd.DataFrame({"node": nodes, "pagerank": vec}).sort_values("pagerank", ascending=False)
                    st.write(f"Δ (L1) to previous = {it['delta_l1']:.6e}; dangling mass carried = {it['dangling_mass']:.6e}")
                    st.dataframe(df, use_container_width=True)

                    # Convergence chart
                    deltas = [itx["delta_l1"] for itx in st.session_state["iterations"]]
                    st.line_chart(pd.DataFrame({"Δ L1": deltas}))
                else:
                    st.warning("No iteration data available for PageRank")

            else:
                if "a" in it and "h" in it:
                    a = it["a"]
                    h = it["h"]
                    df = pd.DataFrame({"node": nodes, "authority": a, "hub": h}).sort_values("authority", ascending=False)
                    st.write(f"Δ (L1) to previous a/h = {it['delta_l1']:.6e}")
                    st.dataframe(df, use_container_width=True)
                    deltas = [itx["delta_l1"] for itx in st.session_state["iterations"]]
                    st.line_chart(pd.DataFrame({"Δ L1": deltas}))
                else:
                    st.warning("No iteration data available for HITS")

        # Matrices
        with st.expander("Matrices", expanded=False):
            G = st.session_state["graph"]
            nodes = list(G.nodes())
            st.write("Adjacency (rows: from, cols: to)")
            A = nx.to_numpy_array(G, nodelist=nodes, dtype=float)
            st.dataframe(pd.DataFrame(A, index=nodes, columns=nodes))
            if st.session_state["algo"] == "PageRank" and len(nodes) > 0:
                # Transition matrix from last run if exists; otherwise compute from current graph
                node_index = {node: i for i, node in enumerate(nodes)}
                P = to_transition_matrix(G, node_index).toarray()
                row_sums = P.sum(axis=1)
                st.write("Row-stochastic transition (dangling rows are zeros; uniform redistribution occurs during iteration)")
                st.dataframe(pd.DataFrame(P, index=nodes, columns=nodes))
                st.write("Row sums:", row_sums)

    with right:
        st.subheader("Current Result")
        if st.session_state["algo"] == "PageRank":
            res = st.session_state.get("result", {})
            if res:
                df_final = pd.DataFrame(sorted(res.items(), key=lambda x: -x[1]), columns=["node", "pagerank"])
                st.dataframe(df_final, use_container_width=True)
                csv = df_final.to_csv(index=False)
                st.download_button("Export final rankings (CSV)", csv, file_name="pagerank_final.csv", mime="text/csv")
        else:
            res = st.session_state.get("result", {})
            if res:
                df_final = pd.DataFrame({
                    "node": list(res["authority"].keys()),
                    "authority": list(res["authority"].values()),
                    "hub": list(res["hub"].values())
                }).sort_values("authority", ascending=False)
                st.dataframe(df_final, use_container_width=True)
                csv = df_final.to_csv(index=False)
                st.download_button("Export final a/h (CSV)", csv, file_name="hits_final.csv", mime="text/csv")

        st.subheader("Guided Check (optional)")
        if st.session_state["iterations"] and st.session_state["algo"] == "PageRank":
            nodes = st.session_state.get("nodes_order", list(st.session_state["graph"].nodes()))
            idx = st.session_state.get("iter_index", 0)
            it = st.session_state["iterations"][idx]
            if "vector" in it:
                node_sel = st.selectbox("Predict next value for node", nodes)
                guess = st.number_input("Your prediction", value=0.0, format="%.6f")
                true_val = float(it["vector"][nodes.index(node_sel)])
                tol_chk = st.number_input("Tolerance", value=1e-3, format="%e")
                if st.button("Check prediction"):
                    ok = abs(guess - true_val) <= tol_chk
                    if ok:
                        st.success("Within tolerance! 🎉")
                    else:
                        st.warning(f"Off by {abs(guess-true_val):.6f}")
                    log(f"Prediction check for {node_sel} at iter {it['k']}: guess={guess} true={true_val}")
            else:
                st.warning("No iteration data available for prediction check")

        st.subheader("Activity Log")
        if st.session_state["activity"]:
            df_log = pd.DataFrame(st.session_state["activity"])  # time, msg
            st.dataframe(df_log, use_container_width=True)
            try:
                md = df_log.to_markdown(index=False)
                st.download_button("Export activity (md)", md, file_name="activity.md", mime="text/markdown")
            except ImportError:
                # Fallback to CSV if tabulate is not available
                csv = df_log.to_csv(index=False)
                st.download_button("Export activity (CSV)", csv, file_name="activity.csv", mime="text/csv")
                st.info("Markdown export requires 'tabulate' package. Using CSV format instead.")


if __name__ == "__main__":
    main()
