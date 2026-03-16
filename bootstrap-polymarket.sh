#!/usr/bin/env bash
set -euo pipefail

# Polymarket bootstrap for Raspberry Pi / Debian-like Linux.
#
# What it does:
# 1) Installs OS dependencies
# 2) Creates a least-privilege service user
# 3) Copies current cloned repo into /opt/polymarket/polymarket-scraper
# 4) Creates Python virtualenv and installs requirements
# 5) Writes /etc/polymarket.env with DATABASE_URL (Supabase Postgres)
# 6) Installs and starts systemd daemon
# 7) Runs DB migrations
#
# Usage examples:
#   DATABASE_URL='postgresql://user:pass@host:5432/postgres?sslmode=require' ./bootstrap-polymarket.sh
#   ./bootstrap-polymarket.sh --database-url 'postgresql://user:pass@host:5432/postgres?sslmode=require' --markets-top-n 50

SERVICE_NAME="polymarket"
SERVICE_USER="polymarket"
INSTALL_BASE="/opt/polymarket"
APP_DIR_NAME="polymarket-scraper"
ENV_FILE="/etc/polymarket.env"
MARKETS_TOP_N="100"
FORCE="false"
DATABASE_URL="${DATABASE_URL:-}"
CLOB_API_KEY="${CLOB_API_KEY:-}"
CLOB_API_KEY_HEADER="${CLOB_API_KEY_HEADER:-X-API-Key}"
CLOB_AUTH_SCHEME="${CLOB_AUTH_SCHEME:-Bearer}"
PROXY_URL="${PROXY_URL:-}"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --database-url URL       Supabase/Postgres DATABASE_URL (required if env var missing)
  --markets-top-n N        Top-N markets for daemon polling (default: 100)
  --service-name NAME      systemd service name (default: polymarket)
  --service-user USER      Service user to run daemon (default: polymarket)
  --install-base PATH      Base install path (default: /opt/polymarket)
  --env-file PATH          Env file path (default: /etc/polymarket.env)
  --force                  Re-copy app and overwrite existing deployment
  -h, --help               Show this help

Env vars also supported:
  DATABASE_URL, CLOB_API_KEY, CLOB_API_KEY_HEADER, CLOB_AUTH_SCHEME, PROXY_URL
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --database-url)
      DATABASE_URL="${2:-}"
      shift 2
      ;;
    --markets-top-n)
      MARKETS_TOP_N="${2:-}"
      shift 2
      ;;
    --service-name)
      SERVICE_NAME="${2:-}"
      shift 2
      ;;
    --service-user)
      SERVICE_USER="${2:-}"
      shift 2
      ;;
    --install-base)
      INSTALL_BASE="${2:-}"
      shift 2
      ;;
    --env-file)
      ENV_FILE="${2:-}"
      shift 2
      ;;
    --force)
      FORCE="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$DATABASE_URL" ]]; then
  if [[ -t 0 ]]; then
    read -r -s -p "Enter DATABASE_URL (Supabase/Postgres): " DATABASE_URL
    echo
  fi
fi

if [[ -z "$DATABASE_URL" ]]; then
  echo "ERROR: DATABASE_URL is required." >&2
  echo "Set env var DATABASE_URL or pass --database-url." >&2
  exit 1
fi

if [[ "$DATABASE_URL" != *"sslmode="* ]]; then
  if [[ "$DATABASE_URL" == *"?"* ]]; then
    DATABASE_URL="${DATABASE_URL}&sslmode=require"
  else
    DATABASE_URL="${DATABASE_URL}?sslmode=require"
  fi
fi

if ! [[ "$MARKETS_TOP_N" =~ ^[0-9]+$ ]] || [[ "$MARKETS_TOP_N" -le 0 ]]; then
  echo "ERROR: --markets-top-n must be a positive integer." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"
TARGET_APP_DIR="$INSTALL_BASE/$APP_DIR_NAME"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
PYTHON_BIN="python3"

if [[ ! -f "$REPO_DIR/main.py" || ! -f "$REPO_DIR/requirements.txt" || ! -d "$REPO_DIR/polymarket_client" ]]; then
  echo "ERROR: Script must be run from cloned polymarket-scraper repo root." >&2
  exit 1
fi

echo "[1/10] Validating sudo access..."
sudo -v

echo "[2/10] Installing OS packages..."
sudo apt-get update
sudo apt-get install -y git "$PYTHON_BIN" python3-venv python3-pip build-essential libpq-dev ca-certificates

echo "[3/10] Creating service user and install base..."
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
  sudo useradd -r -m -d "$INSTALL_BASE" -s /usr/sbin/nologin "$SERVICE_USER"
fi
sudo mkdir -p "$INSTALL_BASE"
sudo chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_BASE"

echo "[4/10] Copying app into $TARGET_APP_DIR ..."
if [[ -d "$TARGET_APP_DIR" ]]; then
  if [[ "$FORCE" != "true" ]]; then
    echo "ERROR: $TARGET_APP_DIR already exists. Re-run with --force to replace it." >&2
    exit 1
  fi
  sudo rm -rf "$TARGET_APP_DIR"
fi
sudo cp -a "$REPO_DIR" "$TARGET_APP_DIR"
sudo chown -R "$SERVICE_USER:$SERVICE_USER" "$TARGET_APP_DIR"

echo "[5/10] Creating virtual environment and installing Python deps..."
sudo -u "$SERVICE_USER" "$PYTHON_BIN" -m venv "$TARGET_APP_DIR/.venv"
sudo -u "$SERVICE_USER" "$TARGET_APP_DIR/.venv/bin/pip" install --upgrade pip
sudo -u "$SERVICE_USER" "$TARGET_APP_DIR/.venv/bin/pip" install -r "$TARGET_APP_DIR/requirements.txt"

echo "[6/10] Writing env file at $ENV_FILE ..."
sudo tee "$ENV_FILE" >/dev/null <<EOF
DATABASE_URL=$DATABASE_URL
EOF

if [[ -n "$CLOB_API_KEY" ]]; then
  echo "CLOB_API_KEY=$CLOB_API_KEY" | sudo tee -a "$ENV_FILE" >/dev/null
  echo "CLOB_API_KEY_HEADER=$CLOB_API_KEY_HEADER" | sudo tee -a "$ENV_FILE" >/dev/null
  echo "CLOB_AUTH_SCHEME=$CLOB_AUTH_SCHEME" | sudo tee -a "$ENV_FILE" >/dev/null
fi

if [[ -n "$PROXY_URL" ]]; then
  echo "PROXY_URL=$PROXY_URL" | sudo tee -a "$ENV_FILE" >/dev/null
fi

sudo chmod 600 "$ENV_FILE"
sudo chown root:root "$ENV_FILE"

echo "[7/10] Installing systemd service at $SERVICE_FILE ..."
sudo tee "$SERVICE_FILE" >/dev/null <<EOF
[Unit]
Description=Polymarket Scraper Daemon
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$TARGET_APP_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$TARGET_APP_DIR/.venv/bin/python main.py daemon --markets-top-n $MARKETS_TOP_N --store
Restart=always
RestartSec=10
KillSignal=SIGINT
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

echo "[8/10] Applying database migrations..."
sudo -u "$SERVICE_USER" bash -lc "cd '$TARGET_APP_DIR' && .venv/bin/python main.py migrate"

echo "[9/10] Enabling and starting service..."
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME.service"
sudo systemctl restart "$SERVICE_NAME.service"

echo "[10/10] Verifying service status..."
sudo systemctl --no-pager --full status "$SERVICE_NAME.service" || true

echo
echo "Bootstrap complete. Useful commands:"
echo "  sudo systemctl status $SERVICE_NAME --no-pager"
echo "  sudo journalctl -u $SERVICE_NAME -f"
echo "  sudo systemctl restart $SERVICE_NAME"
