#!/usr/bin/env python3
"""
Polymarket Analytics Data Pipeline - Entry Point

Usage:
    python main.py sync --entity markets
    python main.py sync --entity trades --condition-id 0x...
    python main.py daemon --markets-top-n 100
    python main.py intelligence sentiment <market_slug>
    python main.py intelligence news <market_slug>
    python main.py intelligence arbitrage
    python main.py intelligence calibration
    python main.py intelligence triggers <market_slug>
    python main.py intelligence pipeline <market_slug>
    python main.py migrate

NOT FINANCIAL ADVICE - For research and educational purposes only.
"""

from cli import app

if __name__ == "__main__":
    app()
