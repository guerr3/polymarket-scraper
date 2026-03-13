"""
CLI interface for the Polymarket Analytics data pipeline.

Usage:
    python cli.py sync --entity markets
    python cli.py sync --entity trades --condition-id 0x...
    python cli.py sync --entity prices --condition-id 0x...
    python cli.py daemon --markets-top-n 100
    python cli.py advisor --condition-id 0x...
    python cli.py backtest --condition-id 0x...
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Optional

import typer

app = typer.Typer(
    name="polymarket",
    help="Polymarket Analytics Data Pipeline & Trading Advisor",
    add_completion=False,
)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# --------------------------------------------------------------------------- #
#  Sync commands
# --------------------------------------------------------------------------- #

@app.command()
def sync(
    entity: str = typer.Option(..., help="Entity to sync: markets, trades, prices, events"),
    condition_id: Optional[str] = typer.Option(None, "--condition-id", "-c", help="Condition ID for trades/prices"),
    store: bool = typer.Option(False, "--store", "-s", help="Store results in PostgreSQL"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output JSON file path"),
    limit: int = typer.Option(100, "--limit", "-l", help="Max items to fetch"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """One-shot sync of markets, trades, or price history."""
    setup_logging(verbose)
    asyncio.run(_sync(entity, condition_id, store, output, limit))


async def _sync(
    entity: str,
    condition_id: Optional[str],
    store: bool,
    output: Optional[str],
    limit: int,
):
    from polymarket_client.resilience import ResilientSession
    from polymarket_client.storage import Storage
    from polymarket_client.realtime import RealtimeUpdater

    async with ResilientSession() as session:
        storage = Storage()
        if store:
            await storage.connect()
            await storage.run_migrations()

        updater = RealtimeUpdater(session, storage)

        if entity == "markets":
            markets = await updater.sync_markets()
            typer.echo(f"Fetched {len(markets)} markets")

            if output:
                data = [m.model_dump(mode="json") for m in markets]
                _write_json(output, data)
                typer.echo(f"Written to {output}")
            else:
                # Print summary
                for m in markets[:20]:
                    prices = ", ".join(
                        f"{o.name}: {o.price:.2%}" for o in m.outcomes
                    )
                    typer.echo(
                        f"  [{m.condition_id[:12]}...] {m.question[:60]} "
                        f"| Vol: ${m.volume:,.0f} | {prices}"
                    )
                if len(markets) > 20:
                    typer.echo(f"  ... and {len(markets) - 20} more")

        elif entity == "trades":
            if not condition_id:
                typer.echo("Error: --condition-id required for trades", err=True)
                raise typer.Exit(1)
            trades = await updater.sync_trades(condition_id)
            typer.echo(f"Fetched {len(trades)} trades for {condition_id[:16]}...")

            if output:
                data = [t.model_dump(mode="json") for t in trades]
                _write_json(output, data)

        elif entity == "prices":
            if not condition_id:
                typer.echo("Error: --condition-id required for prices", err=True)
                raise typer.Exit(1)
            prices = await updater.sync_prices(condition_id)
            typer.echo(f"Fetched {len(prices)} price points for {condition_id[:16]}...")

            if output:
                data = [p.model_dump(mode="json") for p in prices]
                _write_json(output, data)

        elif entity == "events":
            from polymarket_client.gamma_client import GammaClient
            gamma = GammaClient(session)
            events = await gamma.get_all_events(active=True, max_pages=5)
            typer.echo(f"Fetched {len(events)} events")

            if output:
                _write_json(output, events)

        else:
            typer.echo(f"Unknown entity: {entity}. Use: markets, trades, prices, events", err=True)
            raise typer.Exit(1)

        if store:
            await storage.close()


# --------------------------------------------------------------------------- #
#  Daemon mode
# --------------------------------------------------------------------------- #

@app.command()
def daemon(
    markets_top_n: int = typer.Option(100, "--markets-top-n", "-n"),
    store: bool = typer.Option(True, "--store/--no-store"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run continuous real-time data ingestion."""
    setup_logging(verbose)
    typer.echo(f"Starting daemon mode (top {markets_top_n} markets)...")
    asyncio.run(_daemon(markets_top_n, store))


async def _daemon(top_n: int, store: bool):
    from polymarket_client.resilience import ResilientSession
    from polymarket_client.storage import Storage
    from polymarket_client.realtime import RealtimeUpdater

    async with ResilientSession() as session:
        storage = Storage()
        if store:
            await storage.connect()
            await storage.run_migrations()

        updater = RealtimeUpdater(session, storage)
        try:
            await updater.start(top_n=top_n)
        except KeyboardInterrupt:
            await updater.stop()
        finally:
            if store:
                await storage.close()


# --------------------------------------------------------------------------- #
#  Advisor
# --------------------------------------------------------------------------- #

@app.command()
def advisor(
    condition_id: str = typer.Option(..., "--condition-id", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate a trading signal for a specific market. NOT FINANCIAL ADVICE."""
    setup_logging(verbose)
    typer.echo("⚠️  DISCLAIMER: Not financial advice. For research only.\n")
    asyncio.run(_advisor(condition_id))


async def _advisor(condition_id: str):
    from polymarket_client.resilience import ResilientSession
    from polymarket_client.clob_client import ClobClient
    from polymarket_client.gamma_client import GammaClient
    from advisor.signals import generate_signal_from_data

    async with ResilientSession() as session:
        clob = ClobClient(session)

        # Fetch price history
        prices = await clob.get_price_history(condition_id)
        if not prices:
            typer.echo("No price data available for this market.")
            raise typer.Exit(1)

        signal = generate_signal_from_data(
            condition_id=condition_id,
            prices=prices,
        )

        typer.echo(f"Market:     {condition_id}")
        typer.echo(f"Signal:     {signal.signal.value.upper()}")
        typer.echo(f"Confidence: {signal.confidence:.1%}")
        typer.echo(f"Reason:     {signal.reason}")
        typer.echo(f"\nFeatures:")
        for k, v in signal.features.items():
            typer.echo(f"  {k}: {v}")


# --------------------------------------------------------------------------- #
#  Backtest
# --------------------------------------------------------------------------- #

@app.command()
def backtest(
    condition_id: str = typer.Option(..., "--condition-id", "-c"),
    window: int = typer.Option(50, "--window", "-w", help="Lookback window size"),
    hold: int = typer.Option(10, "--hold", help="Hold period (data points)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
):
    """Run a backtest on historical data. NOT FINANCIAL ADVICE."""
    setup_logging(verbose)
    typer.echo("⚠️  DISCLAIMER: Not financial advice. For research only.\n")
    asyncio.run(_backtest(condition_id, window, hold, output))


async def _backtest(condition_id: str, window: int, hold: int, output: Optional[str]):
    from polymarket_client.resilience import ResilientSession
    from polymarket_client.clob_client import ClobClient
    from advisor.backtest import run_backtest, BacktestConfig

    async with ResilientSession() as session:
        clob = ClobClient(session)

        typer.echo(f"Fetching price history for {condition_id[:16]}...")
        prices = await clob.get_price_history(condition_id, fidelity=60)

        if len(prices) < window + hold:
            typer.echo(
                f"Insufficient data: {len(prices)} points "
                f"(need {window + hold})"
            )
            raise typer.Exit(1)

        typer.echo(f"Running backtest on {len(prices)} price points...")

        config = BacktestConfig(
            window_size=window,
            hold_periods=hold,
        )
        result = run_backtest(condition_id, prices, config=config)

        typer.echo(f"\n{'='*50}")
        typer.echo(f"BACKTEST RESULTS")
        typer.echo(f"{'='*50}")
        typer.echo(f"Market:       {condition_id}")
        typer.echo(f"Period:       {result.start_ts} → {result.end_ts}")
        typer.echo(f"Total trades: {result.total_trades}")
        typer.echo(f"Win rate:     {result.win_rate:.1%}")
        typer.echo(f"Total PnL:    ${result.total_pnl:,.2f}")
        typer.echo(f"Max drawdown: ${result.max_drawdown:,.2f}")
        typer.echo(f"Sharpe ratio: {result.sharpe_ratio:.2f}")

        if output:
            _write_json(output, result.model_dump(mode="json"))
            typer.echo(f"\nFull results written to {output}")


# --------------------------------------------------------------------------- #
#  HTML fallback
# --------------------------------------------------------------------------- #

@app.command()
def html_fallback(
    max_pages: int = typer.Option(5, "--max-pages"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Scrape markets from polymarketanalytics.com (fallback only)."""
    setup_logging(verbose)
    asyncio.run(_html_fallback(max_pages, output))


async def _html_fallback(max_pages: int, output: Optional[str]):
    from polymarket_client.html_fallback import HtmlFallbackScraper

    async with HtmlFallbackScraper() as scraper:
        markets = await scraper.get_all_markets(max_pages=max_pages)
        typer.echo(f"Scraped {len(markets)} markets from HTML fallback")

        for m in markets[:10]:
            typer.echo(f"  {m.question[:60]} | Vol: ${m.volume:,.0f}")

        if output:
            data = [m.model_dump(mode="json") for m in markets]
            _write_json(output, data)


# --------------------------------------------------------------------------- #
#  Database migration
# --------------------------------------------------------------------------- #

@app.command()
def migrate(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run database migrations (create tables)."""
    setup_logging(verbose)
    asyncio.run(_migrate())


async def _migrate():
    from polymarket_client.storage import Storage

    storage = Storage()
    await storage.connect()
    await storage.run_migrations()
    await storage.close()
    typer.echo("Migrations applied successfully.")


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _write_json(path: str, data) -> None:
    import json
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


if __name__ == "__main__":
    app()
