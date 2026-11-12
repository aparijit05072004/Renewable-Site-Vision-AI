FROM node:22-slim AS ui
WORKDIR /ui
COPY dashboard-ui/package.json dashboard-ui/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY dashboard-ui/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY terraguard/ terraguard/
COPY artifacts/checkpoints/ artifacts/checkpoints/
COPY --from=ui /ui/dist dashboard-ui/dist
EXPOSE 8000
CMD ["uvicorn", "terraguard.layer4_orchestrator.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
