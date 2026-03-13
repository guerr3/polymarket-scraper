#!/usr/bin/env python3
"""
Polymarket Analytics Data Pipeline - Entry Point

Usage:
    python main.py sync --entity markets
    python main.py sync --entity trades --condition-id 0x...
    python main.py daemon --markets-top-n 100
    python main.py advisor --condition-id 0x...
    python main.py backtest --condition-id 0x...
    python main.py migrate

NOT FINANCIAL ADVICE - For research and educational purposes only.
"""

from cli import app

if __name__ == "__main__":
    app()
