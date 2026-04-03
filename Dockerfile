# Hugging Face Spaces / local Docker
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# OpenEnv Gradio Playground at /web (judges can try Reset → Step in the browser)
ENV ENABLE_WEB_INTERFACE=true

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
