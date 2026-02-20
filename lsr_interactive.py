# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy>=1.24.0",
#     "scikit-learn>=1.3.0",
#     "plotly>=5.18.0",
#     "pandas>=2.0.0",
# ]
# ///

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="LSR: The Diversity Quest")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # The Diversity Quest
        ## *Can You Beat the Algorithm?*

        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); padding: 30px; border-radius: 15px; color: white; text-align: left;">

        <h3 style="margin: 0;">Welcome, Retrieval Engineer!</h3>

        You've been tasked with a critical mission: <b>select the most diverse documents</b>
        from a collapsed embedding neighborhood.<p>

        But beware — the documents are sneaky. They <i>look</i> different but say the same thing.
        Paraphrases everywhere. Redundancy lurking in every corner.<p>

        <b>Your tools: </b>Threshold retrieval, PCA, and your wits.<br
        <b>Your enemy:</b> Density collapse.<br>
        <b>Your goal:</b> Maximum diversity. Minimum redundancy.<p>

        Let the quest begin!<p>

        </div>
        """
    )
    return


@app.cell
def _():
    import numpy as np
    from sklearn.decomposition import PCA
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
    return PCA, go, make_subplots, np, pd, px


@app.cell
def _(mo):
    mo.md(r"""## Chapter 1: Know Your Enemy""")
    return


@app.cell
def _(mo):
    # Fun difficulty selector
    difficulty = mo.ui.radio(
        options=["easy", "medium", "hard", "chaos"],
        value="medium",
        label="Select Your Challenge Level"
    )
    return (difficulty,)


@app.cell
def _(difficulty, mo):
    # Map difficulty to parameters + show callout aligned with radio
    difficulty_settings = {
        "easy": {"collapse": 0.3, "n_docs": 80, "n_clusters": 2, "name": "Novice Retriever"},
        "medium": {"collapse": 0.7, "n_docs": 120, "n_clusters": 3, "name": "Seasoned Searcher"},
        "hard": {"collapse": 0.95, "n_docs": 150, "n_clusters": 4, "name": "Embedding Master"},
        "chaos": {"collapse": 0.99, "n_docs": 200, "n_clusters": 5, "name": "CHAOS MODE"},
    }
    settings = difficulty_settings[difficulty.value]
    collapse_factor = settings["collapse"]

    # Difficulty radio + status side by side
    _intensity = "extremely" if collapse_factor > 0.8 else "moderately" if collapse_factor > 0.5 else "slightly"
    mo.hstack([
        difficulty,
        mo.md(f"""
        <div style="background: {'#fff3cd' if collapse_factor > 0.7 else '#d1ecf1'}; border-left: 4px solid {'#ffc107' if collapse_factor > 0.7 else '#17a2b8'}; padding: 12px 16px; border-radius: 4px;">
        <b>{settings['name']}</b><br>
        Collapse: <b>{collapse_factor:.0%}</b> — {_intensity} clustered
        </div>
        """)
    ], widths=[0.4, 0.6], align="start")
    return collapse_factor, difficulty_settings, settings


@app.cell
def _(mo):
    mo.md(r"""### Fine-Tune Your Parameters""")
    return


@app.cell
def _(mo):
    # Interactive controls
    threshold_slider = mo.ui.slider(
        start=0.5,
        stop=0.95,
        step=0.05,
        value=0.75,
        label="Similarity Threshold (τ) — How picky are you?"
    )

    k_slider = mo.ui.slider(
        start=3,
        stop=20,
        step=1,
        value=8,
        label="Documents to Select (k) — Your retrieval budget"
    )

    seed_input = mo.ui.number(
        start=1,
        stop=9999,
        value=42,
        label="Random Seed — Change to reshuffle the universe"
    )

    mo.vstack([threshold_slider, k_slider, seed_input], align="start")
    return k_slider, seed_input, threshold_slider


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Chapter 2: The Battlefield

        Below lies your embedding space. The colored points are documents above your threshold.
        Notice anything suspicious? *They're all bunched together!*

        This is **density collapse** — your nemesis.
        """
    )
    return


@app.cell
def _(PCA, np, seed_input, settings):
    # Generate synthetic data in 256 dimensions with density collapse
    _EMBED_DIM = 256

    def generate_quest_data(settings, seed):
        """Generate high-dimensional battlefield, project to 3D for display."""
        rng = np.random.RandomState(seed)

        n_docs = settings["n_docs"]
        collapse_factor = settings["collapse"]
        n_clusters = settings["n_clusters"]

        # Query in high-d space
        query = rng.randn(_EMBED_DIM).astype(np.float32)
        query = query / np.linalg.norm(query)

        # Principal direction for collapse
        principal_dir = rng.randn(_EMBED_DIM).astype(np.float32)
        # Orthogonalize against query
        principal_dir = principal_dir - np.dot(principal_dir, query) * query
        principal_dir = principal_dir / np.linalg.norm(principal_dir)

        embeddings = []
        similarities = []
        is_relevant = []
        cluster_ids = []
        doc_names = []

        cluster_themes = [
            ["Alpha", "Prime", "Origin"],
            ["Beta", "Secondary", "Mirror"],
            ["Gamma", "Tertiary", "Echo"],
            ["Delta", "Quaternary", "Shadow"],
            ["Epsilon", "Quinary", "Ghost"],
        ]

        n_relevant = int(n_docs * 0.7)
        docs_per_cluster = max(1, n_relevant // n_clusters)

        # Spread along principal axis vs all other axes
        spread_along = 0.3
        # Perpendicular noise — shrinks with collapse (across ALL 256 dims)
        noise_scale = 0.08 * (1 - collapse_factor * 0.9)

        for i in range(n_docs):
            if i < n_relevant:
                cluster_id = i // docs_per_cluster
                local_id = i % docs_per_cluster

                # Start from query
                doc = query.copy()

                # Spread along principal direction (the manifold)
                cluster_center = (cluster_id - (n_clusters - 1) / 2) * (spread_along / n_clusters)
                offset_along = cluster_center + rng.randn() * (spread_along / (n_clusters * 1.5))
                doc = doc + principal_dir * offset_along

                # Small noise in ALL other dimensions (collapses with factor)
                noise = rng.randn(_EMBED_DIM).astype(np.float32) * noise_scale
                doc = doc + noise

                # Normalize
                doc = doc / np.linalg.norm(doc)

                # Similarity is actual cosine sim to query
                sim = float(np.dot(query, doc))
                is_rel = True
                c_id = cluster_id
                name = f"{cluster_themes[cluster_id % 5][local_id % 3]}-{i:03d}"
            else:
                # Noise: random direction in 256-d
                doc = rng.randn(_EMBED_DIM).astype(np.float32)
                doc = doc / np.linalg.norm(doc)
                sim = float(np.dot(query, doc))
                is_rel = False
                c_id = -1
                name = f"Noise-{i:03d}"

            embeddings.append(doc)
            similarities.append(sim)
            is_relevant.append(is_rel)
            cluster_ids.append(c_id)
            doc_names.append(name)

        embeddings = np.array(embeddings)
        similarities = np.array(similarities)

        # Project to 3D for visualization using PCA on the full corpus
        pca_viz = PCA(n_components=3)
        embeddings_3d = pca_viz.fit_transform(embeddings)
        query_3d = pca_viz.transform(query.reshape(1, -1))[0]
        principal_3d = pca_viz.transform((query + principal_dir * 0.5).reshape(1, -1))[0] - query_3d

        return {
            "embeddings": embeddings,           # 256-d (algorithms use this)
            "embeddings_3d": embeddings_3d,     # 3-d (plots use this)
            "similarities": similarities,
            "is_relevant": np.array(is_relevant),
            "cluster_ids": np.array(cluster_ids),
            "doc_names": doc_names,
            "query": query,
            "query_3d": query_3d,
            "principal_3d": principal_3d,
            "embed_dim": _EMBED_DIM,
        }

    data = generate_quest_data(settings, int(seed_input.value))
    return data, generate_quest_data


@app.cell
def _(data, go, make_subplots, np, threshold_slider):
    # Visualize the battlefield — 3D projection of 256-d data
    threshold = threshold_slider.value
    above_threshold = data["similarities"] >= threshold
    n_above = np.sum(above_threshold)

    # Use 3D projections for display
    _e3d = data["embeddings_3d"]
    _q3d = data["query_3d"]
    _p3d = data["principal_3d"]

    fig_battlefield = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "xy"}]],
        subplot_titles=(
            f"The Battlefield — {data['embed_dim']}d projected to 3d ({n_above} above τ)",
            "Similarity Distribution"
        ),
        column_widths=[0.65, 0.35]
    )

    # Color by cluster for documents above threshold
    colors_above = []
    cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    for _i, _above in enumerate(above_threshold):
        if _above:
            _cid = data["cluster_ids"][_i]
            if _cid >= 0:
                colors_above.append(cluster_colors[_cid % 5])
            else:
                colors_above.append('gray')

    mask_above = above_threshold
    fig_battlefield.add_trace(
        go.Scatter3d(
            x=_e3d[mask_above, 0],
            y=_e3d[mask_above, 1],
            z=_e3d[mask_above, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=colors_above,
                opacity=0.85,
                line=dict(color='white', width=1)
            ),
            name='Targets',
            text=[f"{data['doc_names'][i]}<br>Sim: {data['similarities'][i]:.3f}<br>Cluster: {data['cluster_ids'][i]}"
                  for i in np.where(mask_above)[0]],
            hoverinfo='text'
        ),
        row=1, col=1
    )

    # Below-threshold documents as faint outlines
    mask_below = ~above_threshold
    fig_battlefield.add_trace(
        go.Scatter3d(
            x=_e3d[mask_below, 0],
            y=_e3d[mask_below, 1],
            z=_e3d[mask_below, 2],
            mode='markers',
            marker=dict(
                size=4,
                color='rgba(0,0,0,0)',
                opacity=0.25,
                line=dict(color='rgba(150,150,180,0.4)', width=1)
            ),
            name='Below threshold',
            hovertext=[
                f"{data['doc_names'][i]}<br>Sim: {data['similarities'][i]:.3f}"
                for i in np.where(mask_below)[0]
            ],
            hoverinfo='text'
        ),
        row=1, col=1
    )

    # Query point
    fig_battlefield.add_trace(
        go.Scatter3d(
            x=[_q3d[0]], y=[_q3d[1]], z=[_q3d[2]],
            mode='markers+text',
            marker=dict(size=15, color='gold', symbol='diamond',
                       line=dict(color='black', width=2)),
            text=['QUERY'],
            textposition='top center',
            textfont=dict(size=14, color='gold'),
            name='Query'
        ),
        row=1, col=1
    )

    # Principal direction arrow
    fig_battlefield.add_trace(
        go.Scatter3d(
            x=[_q3d[0], _q3d[0] + _p3d[0]],
            y=[_q3d[1], _q3d[1] + _p3d[1]],
            z=[_q3d[2], _q3d[2] + _p3d[2]],
            mode='lines',
            line=dict(color='orange', width=8),
            name='Collapse Axis'
        ),
        row=1, col=1
    )

    # Similarity histogram with dramatic styling
    fig_battlefield.add_trace(
        go.Histogram(
            x=data["similarities"],
            nbinsx=25,
            marker_color='rgba(65, 105, 225, 0.7)',
            name='All docs'
        ),
        row=1, col=2
    )

    # Add threshold line as a shape (vline doesn't work with mixed subplot types)
    fig_battlefield.add_shape(
        type="line",
        x0=threshold, x1=threshold,
        y0=0, y1=1,
        yref="y domain",
        xref="x2",
        line=dict(color="crimson", width=3, dash="dash"),
    )
    fig_battlefield.add_annotation(
        x=threshold, y=1.05,
        yref="y domain",
        xref="x2",
        text=f"τ = {threshold}",
        showarrow=False,
        font=dict(color="crimson", size=12)
    )

    fig_battlefield.update_layout(
        height=550,
        showlegend=True,
        paper_bgcolor='rgba(20,20,30,0.95)',
        plot_bgcolor='rgba(20,20,30,0.95)',
        font=dict(color='white'),
        scene=dict(
            bgcolor='rgba(20,20,30,0.95)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='X'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Y'),
            zaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Z'),
        )
    )

    fig_battlefield
    return (
        above_threshold,
        cluster_colors,
        colors_above,
        fig_battlefield,
        mask_above,
        mask_below,
        n_above,
        threshold,
    )


@app.cell
def _(cluster_colors, mo, n_above, np, above_threshold, data):
    # Show cluster breakdown
    _cluster_counts = {}
    for _idx, _is_above in enumerate(above_threshold):
        if _is_above:
            _cid = data["cluster_ids"][_idx]
            if _cid >= 0:
                _cluster_counts[_cid] = _cluster_counts.get(_cid, 0) + 1

    cluster_info = " | ".join([
        f"<span style='color:{cluster_colors[c]}'> Cluster {c}: {count}</span>"
        for c, count in sorted(_cluster_counts.items())
    ])

    mo.md(f"""
    ### Intel Report

    **{n_above} documents** detected above threshold.
    {f"Cluster breakdown: {cluster_info}" if _cluster_counts else "No clear clusters detected."}

    **Warning:** Documents in the same cluster are *paraphrases* — selecting multiple
    from one cluster is wasteful redundancy!
    """)
    return (cluster_info,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Chapter 3: Choose Your Weapon

        Time to select your retrieval strategy. Each method has its strengths and weaknesses.
        Choose wisely!
        """
    )
    return


@app.cell
def _(mo):
    # Method selector with descriptions
    method_cards = mo.ui.radio(
        options=["lsr", "topk", "random", "mmr"],
        value="lsr",
        label="Select Your Weapon"
    )
    method_cards
    return (method_cards,)


@app.cell
def _(method_cards, mo):
    # Method descriptions
    _descriptions = {
        "lsr": mo.md("""
        **LSR (The Smart One)** — finds the principal direction of variance using PCA,
        then samples evenly along it. *Smart, efficient, diversity-maximizing.*
        """),
        "topk": mo.md("""
        **Top-k (The Greedy One)** — just grabs the k highest similarity documents. Fast but greedy —
        *it doesn't care about diversity at all!*
        """),
        "random": mo.md("""
        **Random (The Lucky One)** — picks documents at random from those above threshold.
        *Sometimes you get lucky. Usually you don't.*
        """),
        "mmr": mo.md("""
        **MMR (The Balanced One)** — iteratively selects documents that balance
        relevance and diversity. *Good but expensive — O(n·k·d) complexity!*
        """),
    }
    _descriptions[method_cards.value]
    return


@app.cell
def _(
    PCA,
    above_threshold,
    data,
    k_slider,
    method_cards,
    np,
):
    # Implement retrieval methods
    def lsr_retrieve(embeddings, above_mask, k, adaptive=True):
        neighborhood = embeddings[above_mask]
        indices = np.where(above_mask)[0]
        if len(neighborhood) <= k:
            return indices, None, None, {"mode": "passthrough", "reason": "Neighborhood smaller than k"}

        pca = PCA(n_components=min(3, len(neighborhood)))
        pca.fit(neighborhood)
        pc1_variance = pca.explained_variance_ratio_[0]
        projections = pca.transform(neighborhood)[:, 0]

        # Adaptive mode: check if collapse is strong enough for LSR to help
        collapse_info = {
            "pc1_variance": float(pc1_variance),
            "mode": "lsr",
            "reason": "",
        }

        if adaptive:
            if pc1_variance >= 0.70:
                collapse_info["mode"] = "lsr"
                collapse_info["reason"] = f"Strong collapse detected (PC1={pc1_variance:.0%}). LSR is ideal."
            elif pc1_variance >= 0.50:
                collapse_info["mode"] = "lsr"
                collapse_info["reason"] = f"Moderate collapse (PC1={pc1_variance:.0%}). LSR helps, consider 2 components for production."
            else:
                collapse_info["mode"] = "fallback"
                collapse_info["reason"] = f"Weak collapse (PC1={pc1_variance:.0%}). No dominant manifold — MMR or top-k may suffice."

        sorted_idx = np.argsort(projections)
        selected = [sorted_idx[int(i * len(sorted_idx) / k)] for i in range(k)]
        return indices[selected], pca, projections, collapse_info

    def topk_retrieve(similarities, k):
        return np.argsort(-similarities)[:k]

    def random_retrieve(above_mask, k, seed=42):
        rng = np.random.RandomState(seed)
        indices = np.where(above_mask)[0]
        if len(indices) <= k:
            return indices
        return rng.choice(indices, k, replace=False)

    def mmr_retrieve(embeddings, similarities, above_mask, k, lambda_param=0.5):
        candidates = np.where(above_mask)[0]
        if len(candidates) <= k:
            return candidates
        selected = [candidates[np.argmax(similarities[candidates])]]
        while len(selected) < k:
            best_score, best_idx = -np.inf, None
            for idx in candidates:
                if idx in selected:
                    continue
                relevance = similarities[idx]
                redundancy = max(embeddings[idx] @ embeddings[s] for s in selected)
                score = lambda_param * relevance - (1 - lambda_param) * redundancy
                if score > best_score:
                    best_score, best_idx = score, idx
            if best_idx:
                selected.append(best_idx)
            else:
                break
        return np.array(selected)

    # Execute selected method
    k = int(k_slider.value)
    method = method_cards.value

    if method == "lsr":
        selected_indices, pca_model, projections, collapse_info = lsr_retrieve(
            data["embeddings"], above_threshold, k, adaptive=True
        )
    elif method == "topk":
        selected_indices = topk_retrieve(data["similarities"], k)
        pca_model, projections, collapse_info = None, None, None
    elif method == "random":
        selected_indices = random_retrieve(above_threshold, k)
        pca_model, projections, collapse_info = None, None, None
    else:  # mmr
        selected_indices = mmr_retrieve(
            data["embeddings"], data["similarities"], above_threshold, k
        )
        pca_model, projections, collapse_info = None, None, None
    return (
        collapse_info,
        k,
        lsr_retrieve,
        method,
        mmr_retrieve,
        pca_model,
        projections,
        random_retrieve,
        selected_indices,
        topk_retrieve,
    )


@app.cell
def _(collapse_info, collapse_factor, data, method, mo, n_above, np, selected_indices):
    # Reactive explanation — updates whenever method, difficulty, or threshold changes
    _method_names = {"lsr": "LSR", "topk": "Top-k", "random": "Random", "mmr": "MMR"}
    _sel_clusters = set(data["cluster_ids"][i] for i in selected_indices if data["cluster_ids"][i] >= 0)
    _n_clusters_hit = len(_sel_clusters)
    _total_clusters = len(set(data["cluster_ids"][data["cluster_ids"] >= 0]))
    _avg_sim = float(np.mean(data["similarities"][selected_indices]))

    # Build adaptive insight
    _adaptive_note = ""
    if collapse_info and method == "lsr":
        _pc1 = collapse_info.get("pc1_variance", 0)
        _mode = collapse_info.get("mode", "lsr")
        if _pc1 >= 0.70:
            _adaptive_note = f"""
            **Adaptive LSR: Strong collapse detected** (PC1 captures **{_pc1:.0%}** of variance).
            The neighborhood lives on a thin manifold — LSR's quantile sampling is perfectly suited here.
            """
        elif _pc1 >= 0.50:
            _adaptive_note = f"""
            **Adaptive LSR: Moderate collapse** (PC1 captures **{_pc1:.0%}** of variance).
            LSR helps, but a multi-component approach (sampling across PC1 + PC2) could capture more structure.
            """
        else:
            _adaptive_note = f"""
            **Adaptive LSR: Weak collapse** (PC1 captures only **{_pc1:.0%}** of variance).
            The embedding neighborhood is spread across many directions — no dominant manifold.
            In production, you'd fall back to MMR or top-k here. Try increasing the difficulty!
            """

    # Build method-specific insight
    if method == "lsr":
        _method_insight = f"LSR found **{_n_clusters_hit}/{_total_clusters}** clusters by sampling evenly along the principal axis."
    elif method == "topk":
        _method_insight = f"Top-k grabbed the {len(selected_indices)} most similar docs — but hit only **{_n_clusters_hit}/{_total_clusters}** clusters. Greedy similarity chasing misses diversity."
    elif method == "random":
        _method_insight = f"Random sampling found **{_n_clusters_hit}/{_total_clusters}** clusters. Sometimes luck beats strategy... sometimes it doesn't."
    else:
        _method_insight = f"MMR balanced relevance and diversity, hitting **{_n_clusters_hit}/{_total_clusters}** clusters — but at O(n·k·d) cost per query."

    # Build scenario insight (single-hop vs multi-hop)
    if collapse_factor < 0.5:
        _scenario = "At low collapse, this resembles a **single-hop QA** scenario — documents are spread out, and simple top-k often works fine."
    else:
        _scenario = "At high collapse, this mirrors **multi-hop QA** — many paraphrases crowd the neighborhood, and diversity-aware retrieval becomes critical."

    mo.callout(
        mo.md(f"""
        **What's happening right now:**

        You selected **{_method_names[method]}** on a {data['embed_dim']}-dimensional space with **{n_above}** documents above threshold (collapse factor: **{collapse_factor:.0%}**).

        {_method_insight}

        {_adaptive_note}

        {_scenario}
        """),
        kind="success" if _n_clusters_hit >= _total_clusters else "warn" if _n_clusters_hit >= _total_clusters * 0.5 else "danger"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Chapter 4: The Results""")
    return


@app.cell
def _(cluster_colors, data, go, method, np, pca_model, selected_indices):
    # Visualize selection
    fig_selection = go.Figure()

    # Use 3D projections for display
    _e3d = data["embeddings_3d"]
    _q3d = data["query_3d"]

    # All documents faded
    fig_selection.add_trace(
        go.Scatter3d(
            x=_e3d[:, 0],
            y=_e3d[:, 1],
            z=_e3d[:, 2],
            mode='markers',
            marker=dict(size=3, color='gray', opacity=0.2),
            name='All docs',
            hoverinfo='skip'
        )
    )

    # Selected documents with glow effect
    selected_colors = [
        cluster_colors[data["cluster_ids"][i] % 5]
        if data["cluster_ids"][i] >= 0 else 'white'
        for i in selected_indices
    ]

    fig_selection.add_trace(
        go.Scatter3d(
            x=_e3d[selected_indices, 0],
            y=_e3d[selected_indices, 1],
            z=_e3d[selected_indices, 2],
            mode='markers+text',
            marker=dict(
                size=14,
                color=selected_colors,
                symbol='circle',
                line=dict(color='white', width=3)
            ),
            text=[data["doc_names"][i] for i in selected_indices],
            textposition='top center',
            textfont=dict(size=10, color='white'),
            name='SELECTED',
            hoverinfo='text',
            hovertext=[
                f"{data['doc_names'][i]}<br>Sim: {data['similarities'][i]:.3f}<br>Cluster: {data['cluster_ids'][i]}"
                for i in selected_indices
            ]
        )
    )

    # Query
    fig_selection.add_trace(
        go.Scatter3d(
            x=[_q3d[0]], y=[_q3d[1]], z=[_q3d[2]],
            mode='markers',
            marker=dict(size=15, color='gold', symbol='diamond'),
            name='Query'
        )
    )

    fig_selection.update_layout(
        height=500,
        title=dict(
            text=f"Selected {len(selected_indices)} documents using {method.upper()}",
            font=dict(size=18, color='white')
        ),
        paper_bgcolor='rgba(20,20,30,0.95)',
        plot_bgcolor='rgba(20,20,30,0.95)',
        font=dict(color='white'),
        scene=dict(
            bgcolor='rgba(20,20,30,0.95)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            zaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        )
    )

    fig_selection
    return fig_selection, selected_colors


@app.cell
def _(data, mo, np, selected_indices):
    # Calculate scores
    def calc_redundancy(indices):
        if len(indices) < 2:
            return 0.0
        emb = data["embeddings"][indices]
        sims = emb @ emb.T
        np.fill_diagonal(sims, 0)
        n = len(indices)
        return sims.sum() / (n * (n - 1))

    def calc_cluster_diversity(indices):
        """How many different clusters are represented?"""
        clusters = set(data["cluster_ids"][i] for i in indices if data["cluster_ids"][i] >= 0)
        return len(clusters)

    def calc_coverage(indices):
        """Spatial spread of selection."""
        if len(indices) < 2:
            return 0.0
        emb = data["embeddings"][indices]
        return np.std(emb, axis=0).mean()

    redundancy = calc_redundancy(selected_indices)
    cluster_div = calc_cluster_diversity(selected_indices)
    coverage = calc_coverage(selected_indices)
    avg_sim = data["similarities"][selected_indices].mean()

    # Calculate score
    diversity_score = (1 - redundancy) * 40 + cluster_div * 15 + coverage * 30 + avg_sim * 15
    diversity_score = min(100, max(0, diversity_score))

    # Grade
    if diversity_score >= 85:
        grade, emoji, color = "S", "crown", "#FFD700"
    elif diversity_score >= 70:
        grade, emoji, color = "A", "star", "#4ECDC4"
    elif diversity_score >= 55:
        grade, emoji, color = "B", "thumbsup", "#45B7D1"
    elif diversity_score >= 40:
        grade, emoji, color = "C", "thinking", "#FFEAA7"
    else:
        grade, emoji, color = "F", "skull", "#FF6B6B"

    mo.md(f"""
    ## Your Score

    <div style="background: linear-gradient(135deg, {color}22 0%, {color}44 100%);
                border: 3px solid {color}; border-radius: 20px; padding: 30px;
                text-align: center;">

    <h1 style="font-size: 72px; margin: 0; color: {color};">{grade}</h1>
    <h2 style="margin: 10px 0;">{diversity_score:.1f} / 100</h2>

    </div>

    | Metric | Value | Points |
    |--------|-------|--------|
    | Redundancy | {redundancy:.3f} | {(1-redundancy)*40:.1f}/40 (lower=better) |
    | Clusters Hit | {cluster_div} | {cluster_div*15}/? |
    | Spatial Coverage | {coverage:.3f} | {coverage*30:.1f}/30 |
    | Avg Similarity | {avg_sim:.3f} | {avg_sim*15:.1f}/15 |
    """)
    return (
        avg_sim,
        calc_cluster_diversity,
        calc_coverage,
        calc_redundancy,
        cluster_div,
        color,
        coverage,
        diversity_score,
        emoji,
        grade,
        redundancy,
    )


@app.cell
def _(mo):
    mo.md(r"""## The Leaderboard""")
    return


@app.cell
def _(
    above_threshold,
    calc_cluster_diversity,
    calc_coverage,
    calc_redundancy,
    data,
    k,
    lsr_retrieve,
    mmr_retrieve,
    np,
    pd,
    random_retrieve,
    topk_retrieve,
):
    # Compare all methods
    def score_method(indices):
        r = calc_redundancy(indices)
        c = calc_cluster_diversity(indices)
        cov = calc_coverage(indices)
        s = data["similarities"][indices].mean() if len(indices) > 0 else 0
        return min(100, max(0, (1-r)*40 + c*15 + cov*30 + s*15))

    all_methods = {
        "LSR (Quantile)": lsr_retrieve(data["embeddings"], above_threshold, k, adaptive=True)[0],
        "Top-k": topk_retrieve(data["similarities"], k),
        "Random": random_retrieve(above_threshold, k),
        "MMR": mmr_retrieve(data["embeddings"], data["similarities"], above_threshold, k),
    }

    leaderboard = []
    for name, indices in all_methods.items():
        leaderboard.append({
            "Rank": 0,
            "Method": name,
            "Score": score_method(indices),
            "Redundancy": calc_redundancy(indices),
            "Clusters": calc_cluster_diversity(indices),
            "Coverage": calc_coverage(indices),
        })

    leaderboard_df = pd.DataFrame(leaderboard)
    leaderboard_df = leaderboard_df.sort_values("Score", ascending=False).reset_index(drop=True)
    leaderboard_df["Rank"] = ["1st place", "2nd place", "3rd place", "4th place"][:len(leaderboard_df)]
    leaderboard_df
    return all_methods, leaderboard, leaderboard_df, score_method


@app.cell
def _(go, leaderboard_df):
    # Visual leaderboard
    fig_leaderboard = go.Figure()

    colors_lb = ['#FFD700', '#C0C0C0', '#CD7F32', '#888888']

    fig_leaderboard.add_trace(
        go.Bar(
            x=leaderboard_df["Method"],
            y=leaderboard_df["Score"],
            marker_color=colors_lb[:len(leaderboard_df)],
            text=[f"{s:.1f}" for s in leaderboard_df["Score"]],
            textposition='outside',
            textfont=dict(size=16, color='white')
        )
    )

    fig_leaderboard.update_layout(
        title="Method Showdown",
        yaxis_title="Diversity Score",
        height=350,
        paper_bgcolor='rgba(20,20,30,0.95)',
        plot_bgcolor='rgba(20,20,30,0.95)',
        font=dict(color='white'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', range=[0, 110]),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )

    fig_leaderboard
    return colors_lb, fig_leaderboard


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The Science Behind the Magic

        Why does LSR win? Let's look at the eigenvalue spectrum!
        """
    )
    return


@app.cell
def _(PCA, above_threshold, data, go, make_subplots, np):
    # Eigenvalue analysis with drama
    neighborhood = data["embeddings"][above_threshold]

    if len(neighborhood) > 5:
        pca_analysis = PCA(n_components=min(10, len(neighborhood) - 1))
        pca_analysis.fit(neighborhood)

        eigenvalues = pca_analysis.explained_variance_
        explained = pca_analysis.explained_variance_ratio_

        # Calculate collapse ratio
        collapse_ratio = eigenvalues[0] / (eigenvalues[1] + eigenvalues[2] + 1e-8) if len(eigenvalues) > 2 else 0

        fig_eigen = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f"Eigenvalue Spectrum (Collapse Ratio: {collapse_ratio:.1f}x)",
                "Cumulative Variance"
            )
        )

        # Dramatic eigenvalue bars
        bar_colors = ['#FF6B6B'] + ['#4ECDC4'] * (len(eigenvalues) - 1)
        fig_eigen.add_trace(
            go.Bar(
                x=[f"λ{i+1}" for i in range(len(eigenvalues))],
                y=eigenvalues,
                marker_color=bar_colors,
                text=[f"{v:.1%}" for v in explained],
                textposition='outside',
                textfont=dict(color='white')
            ),
            row=1, col=1
        )

        # Cumulative
        cumsum = np.cumsum(explained)
        fig_eigen.add_trace(
            go.Scatter(
                x=[f"λ{i+1}" for i in range(len(cumsum))],
                y=cumsum,
                mode='lines+markers+text',
                marker=dict(size=12, color='#4ECDC4'),
                line=dict(width=3, color='#4ECDC4'),
                fill='tozeroy',
                fillcolor='rgba(78, 205, 196, 0.3)',
                text=[f"{v:.0%}" for v in cumsum],
                textposition='top center',
                textfont=dict(color='white')
            ),
            row=1, col=2
        )

        fig_eigen.add_hline(y=0.9, line_dash="dash", line_color="#FF6B6B",
                           annotation_text="90% threshold", row=1, col=2)

        fig_eigen.update_layout(
            height=400,
            showlegend=False,
            paper_bgcolor='rgba(20,20,30,0.95)',
            plot_bgcolor='rgba(20,20,30,0.95)',
            font=dict(color='white'),
        )
        fig_eigen.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
        fig_eigen.update_yaxes(gridcolor='rgba(255,255,255,0.1)')

        _fig = fig_eigen
    else:
        _fig = go.Figure()
        _fig.add_annotation(text="Need more documents for analysis", showarrow=False)

    _fig
    return (
        bar_colors,
        collapse_ratio,
        cumsum,
        eigenvalues,
        explained,
        fig_eigen,
        neighborhood,
        pca_analysis,
    )


@app.cell
def _(collapse_ratio, mo):
    severity = "EXTREME" if collapse_ratio > 10 else "HIGH" if collapse_ratio > 5 else "MODERATE" if collapse_ratio > 2 else "LOW"
    mo.callout(
        mo.md(f"""
        **Density Collapse Detected: {severity}**

        The first eigenvalue (λ₁) is **{collapse_ratio:.1f}x larger** than λ₂ + λ₃ combined!

        This means your documents are spread along a **1D manifold** in high-dimensional space.
        LSR exploits this by sampling evenly along this manifold.
        """),
        kind="warn" if collapse_ratio > 5 else "info"
    )
    return (severity,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The Real Question: Why Not Just Use MMR?

        The demo above uses **256-dimensional** embeddings (projected to 3D for display).
        MMR is a strong baseline — it explicitly optimizes the relevance-diversity tradeoff.

        But here's the catch: MMR recomputes pairwise similarities **at every selection step**.
        That's O(n·k·d) — fine for small corpora, brutal at scale.

        **Click the button below** to see where LSR's O(|N|·d) complexity pulls ahead.
        """
    )
    return


@app.cell
def _(mo):
    run_benchmark = mo.ui.run_button(label="Run Scaling Benchmark")
    run_benchmark
    return (run_benchmark,)


@app.cell
def _(PCA, go, mo, np, pd, run_benchmark):
    import time as _time

    # Only run when button is clicked
    mo.stop(not run_benchmark.value, mo.md("*Click the button above to run the benchmark...*"))

    def _lsr_bench(embeddings, sims, threshold, k):
        above = sims >= threshold
        neighborhood = embeddings[above]
        indices = np.where(above)[0]
        if len(neighborhood) <= k:
            return indices
        pca = PCA(n_components=1)
        pca.fit(neighborhood)
        proj = pca.transform(neighborhood)[:, 0]
        sorted_idx = np.argsort(proj)
        selected = [sorted_idx[int(i * len(sorted_idx) / k)] for i in range(k)]
        return indices[selected]

    def _mmr_bench(embeddings, sims, threshold, k):
        above = sims >= threshold
        candidates = np.where(above)[0]
        if len(candidates) <= k:
            return candidates
        selected = [candidates[np.argmax(sims[candidates])]]
        while len(selected) < k:
            best_score, best_idx = -np.inf, None
            for idx in candidates:
                if idx in selected:
                    continue
                relevance = sims[idx]
                redundancy = max(float(embeddings[idx] @ embeddings[s]) for s in selected)
                score = 0.5 * relevance - 0.5 * redundancy
                if score > best_score:
                    best_score, best_idx = score, idx
            if best_idx is not None:
                selected.append(best_idx)
            else:
                break
        return np.array(selected)

    dimensions = [3, 32, 128, 512, 1024]
    _n_docs = 500
    _k_bench = 10
    _rng = np.random.RandomState(42)

    bench_results = []

    for dim in dimensions:
        # Generate high-dimensional data with collapse
        _query = _rng.randn(dim).astype(np.float32)
        _query /= np.linalg.norm(_query)
        _pc_dir = _rng.randn(dim).astype(np.float32)
        _pc_dir /= np.linalg.norm(_pc_dir)

        _embs = []
        _sims = []
        for _j in range(_n_docs):
            if _j < 350:
                _doc = _query + _pc_dir * _rng.randn() * 0.3 + _rng.randn(dim) * 0.02
                _doc = _doc / np.linalg.norm(_doc)
                _sims.append(float(np.dot(_query, _doc)))
            else:
                _doc = _rng.randn(dim).astype(np.float32)
                _doc = _doc / np.linalg.norm(_doc)
                _sims.append(float(np.dot(_query, _doc)))
            _embs.append(_doc)

        _embs = np.array(_embs)
        _sims = np.array(_sims)

        # Benchmark LSR
        _start = _time.perf_counter()
        for _ in range(5):
            _lsr_bench(_embs, _sims, 0.5, _k_bench)
        lsr_time = (_time.perf_counter() - _start) / 5 * 1000

        # Benchmark MMR
        _start = _time.perf_counter()
        for _ in range(5):
            _mmr_bench(_embs, _sims, 0.5, _k_bench)
        mmr_time = (_time.perf_counter() - _start) / 5 * 1000

        bench_results.append({
            "Dimensions": dim,
            "LSR (ms)": round(lsr_time, 1),
            "MMR (ms)": round(mmr_time, 1),
            "Speedup": round(mmr_time / max(lsr_time, 0.01), 1),
        })

    bench_df = pd.DataFrame(bench_results)

    # Plot it
    fig_bench = go.Figure()
    fig_bench.add_trace(go.Scatter(
        x=bench_df["Dimensions"], y=bench_df["LSR (ms)"],
        mode='lines+markers', name='LSR',
        line=dict(color='#4ECDC4', width=4),
        marker=dict(size=12)
    ))
    fig_bench.add_trace(go.Scatter(
        x=bench_df["Dimensions"], y=bench_df["MMR (ms)"],
        mode='lines+markers', name='MMR',
        line=dict(color='#FF6B6B', width=4),
        marker=dict(size=12)
    ))

    fig_bench.update_layout(
        title="LSR vs MMR: Time per Query as Dimensions Increase",
        xaxis_title="Embedding Dimensions",
        yaxis_title="Time (ms)",
        height=400,
        paper_bgcolor='rgba(20,20,30,0.95)',
        plot_bgcolor='rgba(20,20,30,0.95)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
    )

    mo.vstack([
        fig_bench,
        bench_df,
        mo.callout(
            mo.md(f"""
            **At 1024 dimensions, LSR is {bench_df.iloc[-1]['Speedup']}x faster than MMR.**

            MMR's cost grows with O(n·k·d) because it recomputes pairwise similarities at every step.
            LSR runs PCA once and samples — O(|N|·d).

            In production (12ms/query overhead), this is the difference between
            real-time and noticeable lag.
            """),
            kind="success"
        )
    ])
    return bench_df, bench_results, fig_bench


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## The Full Picture

        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 25px; border-radius: 15px; color: white;">

        ### What You've Learned

        1. **Density collapse** is real — embeddings cluster along thin manifolds
        2. **PCA reveals structure** — the first eigenvector captures meaningful variation
        3. **LSR exploits this** — sampling along the principal direction maximizes diversity
        4. **MMR is a strong baseline** for diversity — but it's O(n·k·d) expensive
        5. **LSR's edge is efficiency** — comparable diversity at a fraction of the cost, especially at scale

        ### When to use what

        | Scenario | Best Method | Why |
        |----------|------------|-----|
        | **Multi-hop QA** (HotpotQA, complex reasoning) | **LSR** | Needs diverse evidence from multiple sources |
        | **RAG at scale** (1000s of candidates, 1024-d) | **LSR** | O(|N|·d) vs MMR's O(n·k·d) |
        | **Single-hop QA** (SQuAD, factoid) | Top-k or MMR | Less redundancy in results, LSR overhead unnecessary |
        | **Small corpus, low latency** | MMR | Diversity matters, cost is manageable |
        | **Adaptive** (mixed workloads) | **LSR with fallback** | Check PC1 variance → LSR if collapsed, else top-k |

        </div>
        """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md("""
        **Local Spectral Retrieval (LSR)**

        This is the interactive companion to the LSR paper:
        *"Variance-Aware Neighbor Selection via PCA in Thresholded Embedding Neighborhoods"*

        - Paper and source: [github.com/acossette/lsr-research](https://github.com/acossette/lsr-research)
        - Install: `pip install lsr-retrieval`
        - Built with [marimo](https://github.com/marimo-team/marimo) — the reactive Python notebook
        """),
        kind="info"
    )
    return


@app.cell
def _(mo):
    # Easter egg
    mo.accordion({
        "Secret Developer Notes": mo.md("""
        Hey, you found me!

        This notebook was built by **Alison Cossette** as the interactive companion
        to the LSR paper. The gamification makes learning more engaging, the reactive
        widgets let you build intuition through experimentation, and the dark theme
        just looks cool.

        See the full paper and `pip install lsr-retrieval` at:
        [github.com/acossette/lsr-research](https://github.com/acossette/lsr-research)

        *P.S. Try setting the seed to 1337 for a special configuration...*
        """)
    })
    return


if __name__ == "__main__":
    app.run()
