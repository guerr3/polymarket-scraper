"""
CLI interface for the Polymarket Analytics data pipeline.

Usage:
    python cli.py sync --entity markets
    python cli.py sync --entity trades --condition-id 0x...
    python cli.py sync --entity prices --condition-id 0x...
    python cli.py daemon --markets-top-n 100
    python cli.py intelligence sentiment <market_slug>
    python cli.py intelligence news <market_slug>
    python cli.py intelligence arbitrage
    python cli.py intelligence calibration
    python cli.py intelligence triggers <market_slug>
    python cli.py intelligence pipeline <market_slug>

NOT FINANCIAL ADVICE — for research and education only.
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
    help="Polymarket Analytics Data Pipeline & Intelligence",
    add_completion=False,
)

intel_app = typer.Typer(
    name="intelligence",
    help="Intelligence analysis commands. NOT FINANCIAL ADVICE.",
    add_completion=False,
)
app.add_typer(intel_app, name="intelligence")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _write_json(path: str, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


async def _find_market_by_slug(session, slug: str):
    """Fetch markets from GAMMA and find one matching the slug."""
    from polymarket_client.gamma_client import GammaClient

    gamma = GammaClient(session)

    # Direct server-side slug lookup avoids false negatives when the target
    # market is outside the first N paginated active markets.
    market = await gamma.get_market_by_slug(slug)

    markets = await gamma.get_all_markets(active=True, max_pages=10)

    if market:
        # Ensure the target market is included in context-dependent workflows
        # (e.g., arbitrage checks against active market snapshots).
        if not any(m.condition_id == market.condition_id for m in markets):
            markets.append(market)
        return market, markets

    for m in markets:
        if m.slug == slug:
            return m, markets

    # Try inactive markets too
    inactive = await gamma.get_all_markets(active=False, max_pages=5)
    for m in inactive:
        if m.slug == slug:
            return m, markets + inactive

    return None, markets


def _disclaimer():
    typer.echo("NOT FINANCIAL ADVICE — for research and education only.\n")


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
#  Advisor (deprecated — use `intelligence pipeline` instead)
# --------------------------------------------------------------------------- #

@app.command(deprecated=True, hidden=True)
def advisor(
    condition_id: str = typer.Option(..., "--condition-id", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """[Deprecated] Use 'intelligence pipeline' instead. NOT FINANCIAL ADVICE."""
    setup_logging(verbose)
    _disclaimer()
    typer.echo("WARNING: 'advisor' is deprecated. Use 'intelligence pipeline <slug>' instead.\n")
    asyncio.run(_advisor(condition_id))


async def _advisor(condition_id: str):
    from polymarket_client.resilience import ResilientSession
    from polymarket_client.clob_client import ClobClient
    from advisor.signals import generate_signal_from_data

    async with ResilientSession() as session:
        clob = ClobClient(session)

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
#  Backtest (deprecated — kept for backwards compatibility)
# --------------------------------------------------------------------------- #

@app.command(deprecated=True, hidden=True)
def backtest(
    condition_id: str = typer.Option(..., "--condition-id", "-c"),
    window: int = typer.Option(50, "--window", "-w", help="Lookback window size"),
    hold: int = typer.Option(10, "--hold", help="Hold period (data points)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
):
    """[Deprecated] Run a backtest on historical data. NOT FINANCIAL ADVICE."""
    setup_logging(verbose)
    _disclaimer()
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


# =========================================================================== #
#  Intelligence subcommands
# =========================================================================== #

@intel_app.command()
def sentiment(
    market_slug: str = typer.Argument(..., help="Market slug (e.g., 'will-btc-reach-100k')"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output JSON file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run sentiment analysis on news for a market."""
    setup_logging(verbose)
    _disclaimer()
    asyncio.run(_intel_sentiment(market_slug, output))


async def _intel_sentiment(slug: str, output: Optional[str]):
    from polymarket_client.resilience import ResilientSession
    from intelligence.sentiment import KeywordSentimentAnalyzer, aggregate_sentiments
    from intelligence.news_feeds import NewsFeedAggregator

    async with ResilientSession() as session:
        market, _ = await _find_market_by_slug(session, slug)
        if not market:
            typer.echo(f"Market not found: {slug}", err=True)
            raise typer.Exit(1)

        typer.echo(f"Market: {market.question}")
        typer.echo(f"Fetching news and running sentiment analysis...\n")

        analyzer = KeywordSentimentAnalyzer()
        async with NewsFeedAggregator() as news:
            items = await news.get_market_news(market.question, max_age_hours=48)

        typer.echo(f"Found {len(items)} news items")

        if not items:
            typer.echo("No news items found for this market.")
            return

        results = []
        for item in items[:30]:
            result = await analyzer.analyze(
                text=item.full_text,
                context=market.question,
                source=item.source,
            )
            results.append(result)

        agg = aggregate_sentiments(
            results,
            market_condition_id=market.condition_id,
            market_question=market.question,
        )

        typer.echo(f"\nOverall sentiment: {agg.overall_label.value}")
        typer.echo(f"Score:            {agg.overall_score:+.3f}")
        typer.echo(f"Confidence:       {agg.confidence:.1%}")
        typer.echo(f"Sources:          {agg.sample_count}")
        typer.echo(f"Breakdown:        {agg.bullish_count} bullish / "
                   f"{agg.bearish_count} bearish / {agg.neutral_count} neutral")

        if output:
            data = {
                "market_slug": slug,
                "overall_label": agg.overall_label.value,
                "overall_score": agg.overall_score,
                "confidence": agg.confidence,
                "sample_count": agg.sample_count,
                "bullish_count": agg.bullish_count,
                "bearish_count": agg.bearish_count,
                "neutral_count": agg.neutral_count,
            }
            _write_json(output, data)
            typer.echo(f"\nWritten to {output}")


@intel_app.command()
def news(
    market_slug: str = typer.Argument(..., help="Market slug"),
    max_age: int = typer.Option(48, "--max-age", help="Max age of news items in hours"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Fetch and display news feeds for a market."""
    setup_logging(verbose)
    asyncio.run(_intel_news(market_slug, max_age, output))


async def _intel_news(slug: str, max_age: int, output: Optional[str]):
    from polymarket_client.resilience import ResilientSession
    from intelligence.news_feeds import NewsFeedAggregator

    async with ResilientSession() as session:
        market, _ = await _find_market_by_slug(session, slug)
        if not market:
            typer.echo(f"Market not found: {slug}", err=True)
            raise typer.Exit(1)

        typer.echo(f"Market: {market.question}")
        typer.echo(f"Fetching news (max age: {max_age}h)...\n")

        async with NewsFeedAggregator() as news_agg:
            items = await news_agg.get_market_news(market.question, max_age_hours=max_age)

        typer.echo(f"Found {len(items)} news items:\n")
        for i, item in enumerate(items[:25], 1):
            age = f"{item.age_hours:.0f}h ago" if item.age_hours else "unknown"
            typer.echo(f"  {i:2d}. [{item.source}] {item.title[:80]}")
            typer.echo(f"      {age} | {item.url[:60] if item.url else 'no url'}")

        if len(items) > 25:
            typer.echo(f"\n  ... and {len(items) - 25} more")

        if output:
            data = [
                {
                    "title": item.title,
                    "source": item.source,
                    "url": item.url,
                    "published": str(item.published) if item.published else None,
                    "description": item.description[:200],
                }
                for item in items
            ]
            _write_json(output, data)
            typer.echo(f"\nWritten to {output}")


@intel_app.command()
def arbitrage(
    top_n: int = typer.Option(200, "--top-n", "-n", help="Number of markets to scan"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Scan for cross-market arbitrage opportunities."""
    setup_logging(verbose)
    _disclaimer()
    asyncio.run(_intel_arbitrage(top_n, output))


async def _intel_arbitrage(top_n: int, output: Optional[str]):
    from polymarket_client.resilience import ResilientSession
    from polymarket_client.gamma_client import GammaClient
    from intelligence.arbitrage import ArbitrageDetector

    async with ResilientSession() as session:
        gamma = GammaClient(session)
        typer.echo(f"Fetching top {top_n} markets...")
        markets = await gamma.get_all_markets(active=True, max_pages=max(1, top_n // 100))

        typer.echo(f"Scanning {len(markets)} markets for arbitrage...\n")
        detector = ArbitrageDetector()
        opportunities = detector.analyze(markets)

        if not opportunities:
            typer.echo("No arbitrage opportunities detected.")
            return

        typer.echo(f"Found {len(opportunities)} opportunities:\n")
        for i, opp in enumerate(opportunities[:20], 1):
            typer.echo(f"  {i}. [{opp.type.upper()}] {opp.severity}")
            typer.echo(f"     A: {opp.market_a.question[:60]}")
            typer.echo(f"     B: {opp.market_b.question[:60]}")
            typer.echo(f"     Divergence: {opp.divergence:.1%} | Confidence: {opp.confidence:.1%}")
            typer.echo(f"     {opp.reasoning[:80]}")
            typer.echo()

        if output:
            data = [
                {
                    "type": opp.type,
                    "severity": opp.severity,
                    "market_a": opp.market_a.question,
                    "market_b": opp.market_b.question,
                    "price_a": opp.price_a,
                    "price_b": opp.price_b,
                    "divergence": opp.divergence,
                    "confidence": opp.confidence,
                    "reasoning": opp.reasoning,
                }
                for opp in opportunities
            ]
            _write_json(output, data)
            typer.echo(f"Written to {output}")


@intel_app.command()
def calibration(
    top_n: int = typer.Option(200, "--top-n", "-n", help="Number of resolved markets for calibration"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Show calibration analysis — historical accuracy by probability bucket."""
    setup_logging(verbose)
    _disclaimer()
    asyncio.run(_intel_calibration(top_n, output))


async def _intel_calibration(top_n: int, output: Optional[str]):
    from polymarket_client.resilience import ResilientSession
    from polymarket_client.gamma_client import GammaClient
    from intelligence.calibration import CalibrationModel

    async with ResilientSession() as session:
        gamma = GammaClient(session)

        typer.echo("Fetching resolved markets for calibration...")
        resolved = await gamma.get_all_markets(active=False, max_pages=max(1, top_n // 100))

        if not resolved:
            typer.echo("No resolved markets found.")
            raise typer.Exit(1)

        typer.echo(f"Building calibration from {len(resolved)} resolved markets...\n")
        model = CalibrationModel()
        report = model.build_calibration(resolved)

        typer.echo(f"Brier Score:  {report.brier_score:.4f}")
        typer.echo(f"Log Loss:     {report.log_loss:.4f}")
        typer.echo(f"Total Markets: {report.total_markets}")
        typer.echo(f"\nCalibration Curve:")
        typer.echo(f"  {'Bucket':>12}  {'Expected':>8}  {'Actual':>8}  {'Count':>6}  {'Deviation':>10}  {'Exploitable':>11}")
        typer.echo(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*10}  {'-'*11}")

        for b in report.buckets:
            flag = "***" if b.is_exploitable else ""
            typer.echo(
                f"  {b.bin_start:.0%}-{b.bin_end:.0%}"
                f"  {b.expected_rate:>7.1%}"
                f"  {b.actual_rate:>7.1%}"
                f"  {b.market_count:>5}"
                f"  {b.deviation:>+9.1%}"
                f"  {flag:>11}"
            )

        # Show mispriced active markets
        typer.echo("\nChecking active markets for mispricing...")
        active = await gamma.get_all_markets(active=True, max_pages=3)
        signals = model.find_mispriced(active, report)

        if signals:
            typer.echo(f"\nFound {len(signals)} potentially mispriced markets:")
            for sig in signals[:10]:
                typer.echo(
                    f"  {sig.market_question[:50]} | "
                    f"price={sig.current_price:.0%} calibrated={sig.calibrated_probability:.0%} "
                    f"edge={sig.edge:+.1%}"
                )
        else:
            typer.echo("No significantly mispriced markets found.")

        if output:
            data = {
                "brier_score": report.brier_score,
                "log_loss": report.log_loss,
                "total_markets": report.total_markets,
                "buckets": [
                    {
                        "bin_start": b.bin_start,
                        "bin_end": b.bin_end,
                        "expected_rate": b.expected_rate,
                        "actual_rate": b.actual_rate,
                        "market_count": b.market_count,
                        "deviation": b.deviation,
                        "is_exploitable": b.is_exploitable,
                    }
                    for b in report.buckets
                ],
            }
            _write_json(output, data)
            typer.echo(f"\nWritten to {output}")


@intel_app.command()
def triggers(
    market_slug: str = typer.Argument(..., help="Market slug"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Check event-driven triggers for a market."""
    setup_logging(verbose)
    _disclaimer()
    asyncio.run(_intel_triggers(market_slug, output))


async def _intel_triggers(slug: str, output: Optional[str]):
    from polymarket_client.resilience import ResilientSession
    from intelligence.sentiment import KeywordSentimentAnalyzer
    from intelligence.news_feeds import NewsFeedAggregator
    from intelligence.event_triggers import EventTriggerDetector

    async with ResilientSession() as session:
        market, _ = await _find_market_by_slug(session, slug)
        if not market:
            typer.echo(f"Market not found: {slug}", err=True)
            raise typer.Exit(1)

        typer.echo(f"Market: {market.question}")
        typer.echo(f"Scanning for event triggers...\n")

        analyzer = KeywordSentimentAnalyzer()
        async with NewsFeedAggregator() as news_agg:
            items = await news_agg.get_market_news(market.question, max_age_hours=48)

        # Run sentiment so triggers can use it
        results = []
        for item in items[:30]:
            result = await analyzer.analyze(
                text=item.full_text,
                context=market.question,
                source=item.source,
            )
            results.append(result)

        detector = EventTriggerDetector()
        summary = detector.scan(
            news_items=items,
            sentiment_results=results,
            market_question=market.question,
        )

        typer.echo(f"Total triggers: {summary.total_triggers}")
        typer.echo(f"Direction:      {summary.net_direction}")
        typer.echo(
            f"Severity:       {summary.critical_count}C / {summary.high_count}H / "
            f"{summary.medium_count}M / {summary.low_count}L"
        )

        if summary.triggers:
            typer.echo(f"\nTriggers:")
            for i, t in enumerate(summary.triggers, 1):
                actionable = " [ACTIONABLE]" if t.is_actionable else ""
                typer.echo(
                    f"  {i}. [{t.severity.value.upper()}] {t.trigger_type.value} — "
                    f"{t.direction}{actionable}"
                )
                typer.echo(f"     {t.description[:80]}")
                typer.echo(f"     Confidence: {t.confidence:.1%} | Source: {t.source}")
        else:
            typer.echo("\nNo triggers detected.")

        if output:
            data = {
                "market_slug": slug,
                "total_triggers": summary.total_triggers,
                "net_direction": summary.net_direction,
                "triggers": [
                    {
                        "type": t.trigger_type.value,
                        "severity": t.severity.value,
                        "direction": t.direction,
                        "confidence": t.confidence,
                        "description": t.description,
                        "source": t.source,
                    }
                    for t in summary.triggers
                ],
            }
            _write_json(output, data)
            typer.echo(f"\nWritten to {output}")


@intel_app.command()
def pipeline(
    market_slug: str = typer.Argument(..., help="Market slug"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run the full intelligence pipeline on a market (all signals combined)."""
    setup_logging(verbose)
    _disclaimer()
    asyncio.run(_intel_pipeline(market_slug, output))


async def _intel_pipeline(slug: str, output: Optional[str]):
    from polymarket_client.resilience import ResilientSession
    from polymarket_client.gamma_client import GammaClient
    from intelligence.signals import IntelligencePipeline

    async with ResilientSession() as session:
        market, all_markets = await _find_market_by_slug(session, slug)
        if not market:
            typer.echo(f"Market not found: {slug}", err=True)
            raise typer.Exit(1)

        typer.echo(f"Market: {market.question}")
        typer.echo(f"Running full intelligence pipeline...\n")

        # Fetch resolved markets for calibration
        gamma = GammaClient(session)
        resolved = await gamma.get_all_markets(active=False, max_pages=3)

        async with IntelligencePipeline() as pipe:
            report = await pipe.analyze(
                market=market,
                all_markets=all_markets,
                resolved_markets=resolved,
            )

        typer.echo(f"{'='*60}")
        typer.echo(f"  INTELLIGENCE REPORT")
        typer.echo(f"{'='*60}")
        typer.echo(f"  Market:     {report.market_question}")
        typer.echo(f"  Signal:     {report.signal.value.upper()}")
        typer.echo(f"  Score:      {report.composite_score:+.4f}")
        typer.echo(f"  Confidence: {report.confidence:.1%}")
        typer.echo(f"  News items: {report.news_items_analyzed}")
        typer.echo(f"{'='*60}")
        typer.echo(f"\n  Components:")

        for c in report.components:
            typer.echo(
                f"    {c.name:>12}: score={c.score:+.3f}  confidence={c.confidence:.1%}  "
                f"weight={c.weight:.0%}  | {c.detail}"
            )

        if report.arbitrage_opportunities:
            typer.echo(f"\n  Arbitrage opportunities: {len(report.arbitrage_opportunities)}")
            for opp in report.arbitrage_opportunities[:3]:
                typer.echo(f"    - {opp.reasoning[:70]}")

        if report.triggers and report.triggers.total_triggers > 0:
            typer.echo(f"\n  Event triggers: {report.triggers.total_triggers} ({report.triggers.net_direction})")

        typer.echo(f"\n  {report.disclaimer}")

        if output:
            data = {
                "market_question": report.market_question,
                "market_slug": report.market_slug,
                "signal": report.signal.value,
                "composite_score": report.composite_score,
                "confidence": report.confidence,
                "news_items_analyzed": report.news_items_analyzed,
                "components": [
                    {
                        "name": c.name,
                        "score": c.score,
                        "confidence": c.confidence,
                        "weight": c.weight,
                        "detail": c.detail,
                    }
                    for c in report.components
                ],
                "generated_at": str(report.generated_at),
            }
            _write_json(output, data)
            typer.echo(f"\nWritten to {output}")


if __name__ == "__main__":
    app()
