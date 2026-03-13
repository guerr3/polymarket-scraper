FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install playwright browsers (optional, for HTML fallback)
RUN pip install playwright && playwright install --with-deps chromium || true

COPY . .

ENTRYPOINT ["python", "main.py"]
