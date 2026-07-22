FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Быстрая самопроверка при сборке образа: референсные тесты не требуют
# сети и должны проходить всегда.
RUN python -m pytest tests/test_reference.py -q

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
