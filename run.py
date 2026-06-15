#!/usr/bin/env python
"""Stock Screener — main CLI entrypoint.

Usage:
    python run.py ingest       Pull and cache all tickers
    python run.py screen        Run the scoring model
    python run.py thesis        Generate AI theses for shortlist
    python run.py dashboard    Launch Streamlit dashboard
    python run.py all           Run full pipeline
"""

import argparse
import sys
from loguru import logger


def cmd_ingest(args):
    """Pull and cache fundamentals for the full universe."""
    from data.universe import get_universe
    from data.ingest import ingest_universe

    tickers = get_universe()
    logger.info(f"Ingesting {len(tickers)} tickers...")
    n = ingest_universe(tickers, force_refresh=args.force)
    logger.info(f"Done — cached {n}/{len(tickers)} tickers")


def cmd_screen(args):
    """Run the multi-factor scoring model."""
    logger.info("Screening... (not yet implemented)")
    # TODO: Phase 2
    logger.info("Coming in Phase 2 — scoring model")


def cmd_thesis(args):
    """Generate AI analyst theses for the shortlist."""
    from thesis.generate import generate_thesis

    if args.ticker:
        logger.info(f"Generating thesis for {args.ticker}...")
        result = generate_thesis(
            ticker=args.ticker,
            model=args.model,
            max_tokens=args.max_tokens,
        )
        if result["success"]:
            import json
            print(json.dumps(result["report"], indent=2))
            logger.info(
                f"Cost: ${result['cost']:.4f} "
                f"({result['tokens']['input']}+{result['tokens']['output']} tokens)"
            )
        else:
            logger.error(f"Thesis failed: {result['error']}")
    else:
        logger.info("Batch thesis generation coming in next iteration.")
        logger.info("Use --ticker SYMBOL to test with a single ticker.")


def cmd_dashboard(args):
    """Launch the Streamlit dashboard."""
    import subprocess
    from pathlib import Path

    dashboard_path = Path(__file__).parent / "out" / "dashboard.py"
    if not dashboard_path.exists():
        logger.error("Dashboard not found — coming in Phase 4")
        sys.exit(1)

    logger.info(f"Launching dashboard at http://localhost:{args.port}")
    subprocess.run(
        ["streamlit", "run", str(dashboard_path), "--server.port", str(args.port)]
    )


def cmd_notion_sync(args):
    """Sync scored tickers to Notion database."""
    from out.notion_sync import sync_from_cache

    logger.info("Syncing to Notion...")
    result = sync_from_cache(limit=args.limit)
    logger.info(
        f"Done — {result['synced']} synced, "
        f"{result['skipped']} skipped, {result['errors']} errors"
    )


def cmd_all(args):
    """Run the full pipeline: ingest → screen → thesis → dashboard."""
    cmd_ingest(args)
    cmd_screen(args)
    cmd_thesis(args)
    # Don't auto-launch dashboard in 'all' mode — user can do it separately
    logger.info("Pipeline complete. Run 'python run.py dashboard' to view results.")


def main():
    parser = argparse.ArgumentParser(
        description="Stock Screener — DCF valuation and AI analyst thesis generator"
    )
    sub = parser.add_subparsers(dest="command", help="Subcommand")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Pull and cache fundamentals")
    p_ingest.add_argument("--force", action="store_true", help="Force refresh (ignore cache TTL)")

    # screen
    p_screen = sub.add_parser("screen", help="Run the scoring model")
    p_screen.add_argument("--backtest", type=str, help="Run on historical date (YYYY-MM-DD)")

    # thesis
    p_thesis = sub.add_parser("thesis", help="Generate AI theses")
    p_thesis.add_argument("--ticker", type=str, help="Run thesis for a single ticker")
    p_thesis.add_argument("--model", type=str, default="claude-sonnet-4-20250514", help="Anthropic model ID")
    p_thesis.add_argument("--max-tokens", type=int, default=4096, help="Max output tokens")

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Launch Streamlit dashboard")
    p_dash.add_argument("--port", type=int, default=8501, help="Port (default: 8501)")

    # notion-sync
    p_notion = sub.add_parser("notion-sync", help="Sync to Notion database")
    p_notion.add_argument("--limit", type=int, default=None, help="Max tickers to sync")

    # all
    p_all = sub.add_parser("all", help="Run full pipeline")
    p_all.add_argument("--force", action="store_true", help="Force refresh ingest")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "screen":
        cmd_screen(args)
    elif args.command == "thesis":
        cmd_thesis(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)
    elif args.command == "notion-sync":
        cmd_notion_sync(args)
    elif args.command == "all":
        cmd_all(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
