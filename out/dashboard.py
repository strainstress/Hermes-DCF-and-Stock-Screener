"""Streamlit dashboard for the Stock Screener.

Run: streamlit run out/dashboard.py

Pages:
- Home: Ranked table of scored tickers with filters
- Ticker detail: Full thesis, scores, sector comparison
- Run history: Past screening runs
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="Stock Screener Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data Loading ───────────────────────────────────────────────


@st.cache_data(ttl=300)
def load_scored_data() -> pd.DataFrame | None:
    """Load the most recent scored universe from DuckDB."""
    try:
        from data.cache import get_cache
        conn = get_cache()
        df = conn.execute("SELECT * FROM fundamentals ORDER BY market_cap DESC").df()
        if df.empty:
            return None
        return df
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_thesis_reports() -> dict[str, dict]:
    """Load thesis JSON reports from out/reports/."""
    reports_dir = Path(__file__).resolve().parent / "reports"
    if not reports_dir.exists():
        return {}

    reports = {}
    for f in sorted(reports_dir.glob("*.json"), reverse=True):
        try:
            with open(f) as fp:
                data = json.load(fp)
                ticker = data.get("ticker", f.stem.split("_")[0])
                reports[ticker] = data
        except Exception:
            pass
    return reports


# ── Sidebar ────────────────────────────────────────────────────

st.sidebar.title("📊 Stock Screener")
st.sidebar.markdown("Growth → Quality → Momentum")

page = st.sidebar.radio("Navigate", ["🏠 Home", "🔍 Ticker Detail", "📈 Run History"])

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Repo:** [Hermes-DCF-and-Stock-Screener]"
    "(https://github.com/strainstress/Hermes-DCF-and-Stock-Screener)"
)
st.sidebar.markdown("**Docs:** `python run.py --help`")


# ── Home Page ──────────────────────────────────────────────────

if page == "🏠 Home":
    st.title("📊 Stock Screener — Watchlist")

    df = load_scored_data()
    reports = load_thesis_reports()

    if df is None:
        st.warning("No data yet. Run `python run.py screen` first to populate the cache.")
        st.stop()

    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sector_filter = st.multiselect(
            "Sector",
            options=sorted(df["sector"].dropna().unique()),
            default=[],
        )
    with col2:
        min_market_cap = st.number_input(
            "Min Market Cap ($B)",
            min_value=0.0,
            value=5.0,
            step=1.0,
        )
    with col3:
        score_min = st.slider("Min Score", 0, 100, 0)
    with col4:
        search = st.text_input("Search ticker", "", placeholder="e.g., AAPL")

    # Apply filters
    filtered = df.copy()
    if sector_filter:
        filtered = filtered[filtered["sector"].isin(sector_filter)]
    filtered = filtered[filtered["market_cap"] >= min_market_cap * 1e9]

    if search:
        filtered = filtered[
            filtered["ticker"].str.contains(search.upper(), na=False)
            | filtered["name"].str.contains(search, case=False, na=False)
        ]

    # Display metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Tickers", len(df))
    m2.metric("Filtered", len(filtered))
    m3.metric("Reports Ready", len(reports))
    m4.metric("Avg Market Cap", f"${filtered['market_cap'].mean()/1e9:.1f}B")

    st.markdown("---")

    # Build display table
    display_cols = {
        "ticker": "Ticker",
        "name": "Company",
        "sector": "Sector",
        "market_cap": "Market Cap",
    }

    display = filtered[list(display_cols.keys())].copy()
    display.rename(columns=display_cols, inplace=True)
    display["Market Cap"] = display["Market Cap"].apply(
        lambda x: f"${x/1e9:.1f}B" if pd.notna(x) else "N/A"
    )

    # Add thesis verdict if available
    display["Thesis"] = display["Ticker"].apply(
        lambda t: reports.get(t, {}).get("report", {}).get("verdict", "—")
    )
    display["Score"] = display["Ticker"].apply(
        lambda t: f"{reports.get(t, {}).get('report', {}).get('qualitative_scores', {}).get('moat', '—')}"
    )

    st.dataframe(
        display.sort_values("Market Cap", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn(width="small"),
            "Company": st.column_config.TextColumn(width="medium"),
            "Sector": st.column_config.TextColumn(width="small"),
            "Market Cap": st.column_config.TextColumn(width="small"),
            "Thesis": st.column_config.TextColumn(width="small"),
            "Score": st.column_config.TextColumn(width="small"),
        },
    )


# ── Ticker Detail ──────────────────────────────────────────────

elif page == "🔍 Ticker Detail":
    st.title("🔍 Ticker Detail")

    reports = load_thesis_reports()
    df = load_scored_data()

    # Ticker selector
    tickers = sorted(set(list(reports.keys()) + (df["ticker"].tolist() if df is not None else [])))
    selected = st.selectbox("Select Ticker", tickers) if tickers else None

    if not selected:
        st.info("Select a ticker to view its thesis and financials.")
        st.stop()

    # Company info
    if df is not None:
        row = df[df["ticker"] == selected]
        if not row.empty:
            r = row.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Market Cap", f"${r['market_cap']/1e9:.1f}B")
            c2.metric("Revenue (TTM)", f"${r['revenue_ttm']/1e9:.1f}B")
            c3.metric("FCF Margin", f"{r['fcf_margin']*100:.1f}%")
            c4.metric("ROE", f"{r['roe']*100:.1f}%")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Gross Margin", f"{r['gross_margin']*100:.1f}%")
            c6.metric("Debt/Equity", f"{r['debt_equity']:.2f}")
            c7.metric("Altman Z", f"{r['altman_z']:.2f}")
            c8.metric("Interest Coverage", f"{r['interest_coverage']:.1f}x")

    st.markdown("---")

    # Thesis report
    report = reports.get(selected, {})
    thesis = report.get("report", {})

    if thesis:
        st.subheader("📝 AI Analyst Thesis")

        # Verdict badge
        verdict = thesis.get("verdict", "N/A")
        verdict_color = {
            "Strong Buy": "🟢",
            "Buy": "🟢",
            "Watch": "🟡",
            "Pass": "🔴",
            "Sell": "🔴",
        }
        st.markdown(
            f"### Verdict: {verdict_color.get(verdict, '⚪')} **{verdict}** "
            f"(Confidence: {thesis.get('confidence', 0)}%)"
        )

        st.markdown(f"**TL;DR:** {thesis.get('tldr', 'No summary available.')}")

        col_bull, col_bear = st.columns(2)
        with col_bull:
            st.markdown("#### 🐂 Bull Case")
            for point in thesis.get("bull_case", []):
                st.markdown(f"- {point}")

        with col_bear:
            st.markdown("#### 🐻 Bear Case")
            for point in thesis.get("bear_case", []):
                st.markdown(f"- {point}")

        # Qualitative scores
        scores = thesis.get("qualitative_scores", {})
        if scores:
            st.markdown("#### 📊 Qualitative Scores")
            cols = st.columns(4)
            score_labels = {
                "moat": "Moat",
                "management": "Management",
                "capital_allocation": "Cap. Allocation",
                "industry_position": "Industry Position",
            }
            for i, (key, label) in enumerate(score_labels.items()):
                val = scores.get(key, 0)
                cols[i].metric(label, f"{val}/10")

        # Key metrics to watch
        metrics = thesis.get("key_metrics_to_watch", [])
        if metrics:
            st.markdown("#### 👀 Key Metrics to Watch")
            for m in metrics:
                st.markdown(f"- {m}")

        # Peer comparison
        peer = thesis.get("peer_comparison", "")
        if peer:
            st.markdown("#### 🏢 Peer Comparison")
            st.markdown(peer)

        # Sources cited
        sources = thesis.get("sources_cited", [])
        if sources:
            st.markdown("#### 📚 Sources Cited")
            for s in sources:
                st.markdown(f"- {s}")

        st.markdown("---")
        st.caption(
            f"Report generated via Claude Sonnet. "
            f"Cost: ${report.get('cost', 0):.4f} | "
            f"Tokens: {report.get('tokens', {}).get('input', 0)}+{report.get('tokens', {}).get('output', 0)}"
        )
    else:
        st.info(f"No thesis report yet for {selected}. Run `python run.py thesis --ticker {selected}` to generate one.")


# ── Run History ────────────────────────────────────────────────

elif page == "📈 Run History":
    st.title("📈 Run History")

    reports_dir = Path(__file__).resolve().parent / "reports"

    if not reports_dir.exists() or not list(reports_dir.glob("*.json")):
        st.info("No past runs found. Reports will appear here after running `python run.py thesis`.")
    else:
        files = sorted(reports_dir.glob("*.json"), reverse=True)
        st.markdown(f"**{len(files)} reports** generated so far.")

        for f in files[:20]:  # Show last 20
            try:
                with open(f) as fp:
                    data = json.load(fp)
                ticker = data.get("ticker", "???")
                report = data.get("report", {})
                verdict = report.get("verdict", "N/A")
                tldr = report.get("tldr", "")[:120]
                ts = datetime.fromtimestamp(f.stat().st_mtime)

                with st.expander(f"{ticker} — {verdict} ({ts.strftime('%Y-%m-%d %H:%M')})"):
                    st.markdown(f"**{tldr}...**")
                    st.caption(f"Cost: ${data.get('cost', 0):.4f}")
            except Exception:
                pass
