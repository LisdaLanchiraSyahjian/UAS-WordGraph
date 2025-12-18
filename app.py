import streamlit as st

# HARUS PALING ATAS
st.set_page_config(
    page_title="Word Graph & Centrality",
    layout="wide"
)

import PyPDF2
import re
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

st.title("📄 Word Graph & Centrality dari Jurnal PDF")

uploaded_file = st.file_uploader("📤 Upload Jurnal (PDF)", type="pdf")

if uploaded_file:

    # =============================
    # 1. LOAD & EXTRACT TEXT
    # =============================
    text = ""
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        text += (page.extract_text() or "") + " "

    # =============================
    # 2. PREPROCESSING
    # =============================
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    tokens = text.split()

    stopwords = {
        "dan","yang","dari","ke","di","pada","untuk","dengan","adalah",
        "the","of","to","in","for","is","are","that","this","as","by"
    }

    tokens = [w for w in tokens if w not in stopwords and len(w) > 2]

    st.success(f"Total token setelah preprocessing: {len(tokens)}")

    # =============================
    # 3. WORD CO-OCCURRENCE GRAPH
    # =============================
    edges = defaultdict(int)

    for i in range(len(tokens)-1):
        a, b = tokens[i], tokens[i+1]
        if a != b:
            edges[tuple(sorted((a, b)))] += 1

    G = nx.Graph()
    for (a, b), w in edges.items():
        G.add_edge(a, b, weight=w)

    st.info(f"Jumlah node: {G.number_of_nodes()} | Jumlah edge: {G.number_of_edges()}")

    # =============================
    # 4. CENTRALITY MEASURES
    # =============================
    degree_c = nx.degree_centrality(G)
    betweenness_c = nx.betweenness_centrality(G, weight="weight")
    closeness_c = nx.closeness_centrality(G)
    eigenvector_c = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
    pagerank_c = nx.pagerank(G, weight="weight")

    # =============================
    # 5. CENTRALITY DATAFRAME
    # =============================
    df_centrality = pd.DataFrame({
        "Word": list(G.nodes()),
        "Degree": [degree_c[w] for w in G.nodes()],
        "Betweenness": [betweenness_c[w] for w in G.nodes()],
        "Closeness": [closeness_c[w] for w in G.nodes()],
        "Eigenvector": [eigenvector_c[w] for w in G.nodes()],
        "PageRank": [pagerank_c[w] for w in G.nodes()],
    })

    df_centrality = df_centrality.sort_values("PageRank", ascending=False)

    st.subheader("📊 Centrality Measures (Top 20)")
    st.dataframe(df_centrality.head(20), use_container_width=True)

    # =============================
    # 6. CO-OCCURRENCE TABLE
    # =============================
    df_edges = pd.DataFrame(
        [(a, b, w) for (a, b), w in edges.items()],
        columns=["Word 1", "Word 2", "Co-occurrence"]
    ).sort_values("Co-occurrence", ascending=False)

    st.subheader("🔗 Co-occurrence Words (Top 20)")
    st.dataframe(df_edges.head(20), use_container_width=True)

    # =============================
    # 7. VISUALISASI WORD GRAPH
    # =============================
    st.subheader("🕸 Word Graph (Node size = PageRank)")

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.15, seed=42)

    node_sizes = [pagerank_c[n] * 10000 for n in G.nodes()]

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color="skyblue",
        alpha=0.85,
        ax=ax
    )
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)

    top_words = dict(df_centrality.head(10)[["Word", "PageRank"]].values)
    nx.draw_networkx_labels(G, pos, top_words, font_size=10, ax=ax)

    ax.axis("off")
    st.pyplot(fig)
