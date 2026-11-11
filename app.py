"""
app.py
------
Phase 7 — OlympiadMind Streamlit Frontend

4 pages:
  1. Problem Solver  — solve physics problems with full pipeline
  2. Graph Explorer  — browse Neo4j graph with pyvis visualization
  3. Extraction Dashboard — quality metrics and extraction stats
  4. Benchmark Results   — OlympiadMind vs zero-shot comparison

Run with: streamlit run app.py
"""

import json
import os
import sys
import time
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "OlympiadMind",
    page_icon  = "⚛️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── sidebar navigation ────────────────────────────────────────────────────────

st.sidebar.title("⚛️ OlympiadMind")
st.sidebar.caption("Physics Problem Intelligence System")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🧪 Problem Solver", "🕸️ Graph Explorer", "📊 Extraction Dashboard", "🏆 Benchmark Results"],
)

st.sidebar.divider()
st.sidebar.caption("**Stack:** RAPTOR · Neo4j · FAISS · Mistral 7B · LangGraph")


# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource
def get_neo4j_driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )


def run_agent(problem_text: str) -> dict:
    from src.agent.agent import solve
    return solve(problem_text)


# ── page 1: problem solver ────────────────────────────────────────────────────

if page == "🧪 Problem Solver":
    st.title("🧪 Physics Problem Solver")
    st.caption("Powered by RAPTOR + Neo4j + FAISS hybrid retrieval · Mistral 7B generation")

    col1, col2 = st.columns([2, 1])

    with col1:
        problem_text = st.text_area(
            "Enter a physics problem:",
            value=st.session_state.get("problem_input", ""),
            height=150,
            placeholder="A charged particle of mass m and charge q enters a uniform magnetic field B...",
        )

        example_problems = [
            "A charged particle of mass m and charge q enters a uniform magnetic field B perpendicular to the field with velocity v. Find the radius and time period.",
            "A simple pendulum of length L oscillates with small amplitude. Find the time period and discuss the effect of temperature on it.",
            "Two parallel plates of area A are separated by distance d. A dielectric of constant K is inserted. Find the new capacitance.",
        ]

        with st.expander("📝 Example problems"):
            for i, ex in enumerate(example_problems):
                if st.button(f"Example {i+1}", key=f"ex_{i}"):
                    st.session_state.problem_input = ex
                    st.rerun()

    with col2:
        st.markdown("**Pipeline:**")
        st.markdown("""
        1. 🏷️ Classify topic
        2. 🔍 Extract concepts
        3. 🌳 RAPTOR retrieval
        4. 🕸️ Neo4j graph search
        5. 📚 FAISS corpus search
        6. 🤖 Mistral 7B generation
        7. ✅ Validate solution
        """)

    if st.button("⚡ Solve Problem", type="primary", disabled=not problem_text.strip()):
        with st.spinner("Running full pipeline (60-90 seconds)..."):
            start  = time.time()
            result = run_agent(problem_text)
            elapsed = time.time() - start

        st.success(f"Solved in {elapsed:.1f}s")

        # metadata row
        m1, m2, m3 = st.columns(3)
        m1.metric("Topic",      result.get("topic", "Unknown"))
        m2.metric("Confidence", f"{result.get('confidence', 0):.0%}")
        m3.metric("Latency",    f"{elapsed:.1f}s")

        # concepts
        concepts = result.get("concepts", [])
        if concepts:
            st.markdown("**Concepts identified:**")
            st.write(" · ".join(f"`{c}`" for c in concepts))

        
        st.markdown("---")
        st.markdown("### Solution")
        import re
        raw_solution = result.get("solution", "No solution generated.")
        clean_solution = re.sub(r'[✓√].+?\n', '', raw_solution)
        clean_solution = re.sub(r'Loading weights:.+?\n', '', clean_solution)
        clean_solution = re.sub(r'Fetching \d+ files:.+?\n', '', clean_solution)
        clean_solution = re.sub(r'\[1A.+?\n', '', clean_solution)
        clean_solution = re.sub(r'Setting pad_token_id.+?\n', '', clean_solution)
        clean_solution = clean_solution.strip()
        st.markdown(clean_solution)
        if result.get("trace_url"):
            st.markdown("---")
            st.markdown(f"🔍 [View LangSmith Trace]({result['trace_url']})", unsafe_allow_html=True)



        # citations
        citations = result.get("citations", [])
        if citations:
            with st.expander(f"📎 Citations ({len(citations)})"):
                for c in citations:
                    st.markdown(f"- {c}")


# ── page 2: graph explorer ────────────────────────────────────────────────────

elif page == "🕸️ Graph Explorer":
    st.title("🕸️ Knowledge Graph Explorer")
    st.caption("Browse 1,990 nodes and 3,065 relationships in Neo4j")

    driver = get_neo4j_driver()

    tab1, tab2, tab3 = st.tabs(["Search Problems", "Concept Network", "Graph Stats"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        source      = col1.selectbox("Source", ["All", "olympiad", "cbse", "science_bowl", "hindi"])
        competition = col2.text_input("Competition", placeholder="IPhO, APhO, NBPhO...")
        year        = col3.text_input("Year", placeholder="2022")

        if st.button("🔍 Search"):
            filters = []
            params  = {"limit": 20}

            if source != "All":
                filters.append("p.source = $source")
                params["source"] = source
            if competition:
                filters.append("p.competition = $competition")
                params["competition"] = competition
            if year:
                filters.append("p.year = $year")
                params["year"] = year

            where = ("WHERE " + " AND ".join(filters)) if filters else ""
            query = f"""
                MATCH (p:Problem) {where}
                RETURN p.id AS id, p.source AS source,
                       p.competition AS competition, p.year AS year,
                       p.equation_count AS equations, p.figure_count AS figures
                LIMIT $limit
            """

            with driver.session() as session:
                records = [dict(r) for r in session.run(query, **params)]

            st.dataframe(records, use_container_width=True)

            if records:
                selected = st.selectbox("Select problem to explore", [r["id"] for r in records])
                if selected:
                    with driver.session() as session:
                        concepts = [r["concept"] for r in session.run(
                            "MATCH (p:Problem {id: $id})-[:REQUIRES]->(c:Concept) RETURN c.canonical_name AS concept",
                            id=selected
                        )]
                        similar = [dict(r) for r in session.run(
                            "MATCH (p:Problem {id: $id})-[r:SIMILAR_TO]-(s:Problem) RETURN s.id AS id, r.shared_concepts AS shared ORDER BY shared DESC LIMIT 5",
                            id=selected
                        )]

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Concepts:**")
                        for c in concepts:
                            st.markdown(f"- `{c}`")
                    with c2:
                        st.markdown("**Similar problems:**")
                        for s in similar:
                            st.markdown(f"- {s['id']} (shared: {s['shared']})")

    with tab2:
        st.markdown("**Concept network — top connected concepts**")

        with driver.session() as session:
            result = session.run("""
                MATCH (c:Concept)<-[:REQUIRES]-(p:Problem)
                WITH c, count(p) AS n_problems
                ORDER BY n_problems DESC LIMIT 30
                MATCH (c)<-[:REQUIRES]-(p:Problem)-[:REQUIRES]->(c2:Concept)
                WHERE c <> c2
                RETURN c.canonical_name AS concept1,
                       c2.canonical_name AS concept2,
                       count(p) AS weight
                LIMIT 100
            """)
            edges = [dict(r) for r in result]

        if edges:
            from pyvis.network import Network
            import streamlit.components.v1 as components

            net = Network(height="500px", width="100%", bgcolor="#0e1117", font_color="white")
            net.set_options('{"physics": {"stabilization": {"iterations": 100}}}')

            nodes_added = set()
            for e in edges:
                if e["concept1"] not in nodes_added:
                    net.add_node(e["concept1"], label=e["concept1"][:20], color="#4e9af1")
                    nodes_added.add(e["concept1"])
                if e["concept2"] not in nodes_added:
                    net.add_node(e["concept2"], label=e["concept2"][:20], color="#4e9af1")
                    nodes_added.add(e["concept2"])
                net.add_edge(e["concept1"], e["concept2"], value=e["weight"])

            net.save_graph("graph.html")
            with open("graph.html", encoding="utf-8") as f:
                html = f.read()
            components.html(html, height=520)
        else:
            st.info("No concept network data available.")

    with tab3:
        with driver.session() as session:
            node_counts = {r["label"]: r["count"] for r in session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count"
            )}
            rel_counts = {r["type"]: r["count"] for r in session.run(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count"
            )}
            source_counts = {r["source"]: r["count"] for r in session.run(
                "MATCH (p:Problem) RETURN p.source AS source, count(p) AS count"
            )}

        c1, c2, c3 = st.columns(3)
        c1.metric("Problem nodes",  node_counts.get("Problem", 0))
        c2.metric("Concept nodes",  node_counts.get("Concept", 0))
        c3.metric("Relationships",  sum(rel_counts.values()))

        fig = px.bar(
            x=list(source_counts.keys()),
            y=list(source_counts.values()),
            title="Problems by Source",
            labels={"x": "Source", "y": "Count"},
            color=list(source_counts.values()),
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True)


# ── page 3: extraction dashboard ──────────────────────────────────────────────

elif page == "📊 Extraction Dashboard":
    st.title("📊 Extraction Dashboard")
    st.caption("Per-modality quality metrics across 1120 PDFs")

    # quality report
    quality_path = Path("data/evaluation/quality_report.json")
    if quality_path.exists():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))

        st.markdown("### Quality Metrics")
        cols = st.columns(4)
        metrics = [
            ("Typeset text", "text",      "CER",       True),
            ("Equations",    "equations", "chrF",      False),
            ("Scanned OCR",  "scanned",   "WER",       True),
            ("Figures",      "figures",   "BERTScore", False),
        ]

        for col, (label, key, metric, lower_is_better) in zip(cols, metrics):
            r     = quality.get(key, {})
            score = r.get("score", 0)
            target = r.get("target", 0)
            passed = r.get("pass", False)

            if lower_is_better:
                display = f"{score*100:.1f}%"
            else:
                display = f"{score:.3f}"

            col.metric(
                label    = f"{label} ({metric})",
                value    = display,
                delta    = "PASS ✓" if passed else "FAIL ✗",
                delta_color = "normal" if passed else "inverse",
            )

    # extraction stats
    st.markdown("---")
    st.markdown("### Extraction Statistics")

    stats = {
        "Text pages extracted":    1117,
        "Equations extracted":     29435,
        "Figures extracted":       3743,
        "Figures described":       3274,
        "Scanned pages OCRed":     66,
        "Documents merged":        1096,
    }

    s1, s2, s3 = st.columns(3)
    items = list(stats.items())
    for i, (k, v) in enumerate(items):
        col = [s1, s2, s3][i % 3]
        col.metric(k, f"{v:,}")

    # source breakdown
    st.markdown("---")
    st.markdown("### Source Breakdown")

    sources = {
        "olympiad":     1113,
        "cbse":         1,
        "science_bowl": 1,
        "hindi":        2,
        "multilingual": 1,
        "scanned":      1,
        "tabular":      1,
    }

    fig = px.pie(
        values = list(sources.values()),
        names  = list(sources.keys()),
        title  = "Documents by Source",
        color_discrete_sequence = px.colors.sequential.Blues_r,
    )
    st.plotly_chart(fig, use_container_width=True)

    # sample equations
    st.markdown("---")
    st.markdown("### Sample Extracted Equations")

    sample_equations = [
        r"\frac{mv^2}{r} = qvB",
        r"T = \frac{2\pi m}{qB}",
        r"\oint \vec{E} \cdot d\vec{A} = \frac{Q}{\epsilon_0}",
        r"\Delta G = \Delta H - T\Delta S",
        r"E = mc^2",
    ]

    cols = st.columns(len(sample_equations))
    for col, eq in zip(cols, sample_equations):
        col.latex(eq)


# ── page 4: benchmark results ─────────────────────────────────────────────────

elif page == "🏆 Benchmark Results":
    st.title("🏆 Benchmark Results")
    st.caption("OlympiadMind (RAG) vs Zero-Shot Mistral 7B · 37 physics problems")

    benchmark_path = Path("data/evaluation/benchmark_results.json")

    if not benchmark_path.exists():
        st.info("Benchmark not yet complete. Run `python src/evaluation/benchmark.py` to generate results.")
    else:
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        results   = benchmark.get("results", [])
        n         = len(results)

        if results:
            def avg(key, pipeline):
                vals = [r[pipeline].get(key, 0) for r in results if r.get(pipeline)]
                return sum(vals) / len(vals) if vals else 0

            def pct(key, pipeline):
                vals = [r[pipeline].get(key, False) for r in results if r.get(pipeline)]
                return sum(vals) / len(vals) * 100 if vals else 0

            om_ans  = pct("has_answer",    "olympiadmind")
            zs_ans  = pct("has_answer",    "zero_shot")
            om_eq   = pct("has_equations", "olympiadmind")
            zs_eq   = pct("has_equations", "zero_shot")
            om_step = pct("has_steps",     "olympiadmind")
            zs_step = pct("has_steps",     "zero_shot")
            om_lat  = avg("latency",       "olympiadmind")
            zs_lat  = avg("latency",       "zero_shot")

            # metric cards
            col1, col2, col3 = st.columns(3)
            col1.metric("Has Answer",    f"{om_ans:.1f}%", f"+{om_ans - zs_ans:.1f}% vs zero-shot")
            col2.metric("Has Equations", f"{om_eq:.1f}%",  f"+{om_eq  - zs_eq:.1f}% vs zero-shot")
            col3.metric("Avg Latency",   f"{om_lat:.1f}s", f"{om_lat  - zs_lat:+.1f}s vs zero-shot")

            st.markdown("---")

            # bar chart comparison
            metric_names = ["Has Equations", "Has Steps", "Has Answer"]
            om_vals = [om_eq, om_step, om_ans]
            zs_vals = [zs_eq, zs_step, zs_ans]

            fig = go.Figure(data=[
                go.Bar(name="OlympiadMind (RAG)",   x=metric_names, y=om_vals,
                       marker_color="#4F8BF9",
                       text=[f"{v:.1f}%" for v in om_vals], textposition="outside"),
                go.Bar(name="Zero-Shot Mistral 7B", x=metric_names, y=zs_vals,
                       marker_color="#F94F4F",
                       text=[f"{v:.1f}%" for v in zs_vals], textposition="outside"),
            ])
            fig.update_layout(
                barmode="group",
                title="OlympiadMind vs Zero-Shot — Solution Quality",
                yaxis=dict(range=[0, 115], title="Score (%)"),
                legend=dict(orientation="h", y=-0.2),
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # per problem solution length
            doc_ids = [r["doc_id"] for r in results]
            om_lens = [r["olympiadmind"].get("solution_length", 0) for r in results]
            zs_lens = [r["zero_shot"].get("solution_length", 0)    for r in results]

            fig2 = go.Figure(data=[
                go.Scatter(x=doc_ids, y=om_lens, mode="lines+markers",
                           name="OlympiadMind", line=dict(color="#4F8BF9")),
                go.Scatter(x=doc_ids, y=zs_lens, mode="lines+markers",
                           name="Zero-Shot",    line=dict(color="#F94F4F")),
            ])
            fig2.update_layout(
                title="Solution Length per Problem",
                yaxis_title="Characters",
                xaxis_tickangle=-45,
                height=400,
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("---")

            # per-problem table
            with st.expander("📋 Per-problem results"):
                table = []
                for r in results:
                    table.append({
                        "Problem":      r["doc_id"],
                        "OM equations": "✓" if r.get("olympiadmind", {}).get("has_equations") else "✗",
                        "OM steps":     "✓" if r.get("olympiadmind", {}).get("has_steps")     else "✗",
                        "OM answer":    "✓" if r.get("olympiadmind", {}).get("has_answer")    else "✗",
                        "ZS equations": "✓" if r.get("zero_shot",    {}).get("has_equations") else "✗",
                        "ZS steps":     "✓" if r.get("zero_shot",    {}).get("has_steps")     else "✗",
                        "ZS answer":    "✓" if r.get("zero_shot",    {}).get("has_answer")    else "✗",
                    })
                st.dataframe(table, use_container_width=True)

            st.markdown(f"**Problems evaluated:** {n} &nbsp;|&nbsp; "
                        f"**Headline result:** Has Answer +{om_ans - zs_ans:.1f}% over zero-shot")