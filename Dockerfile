# Dockerfile (python web service)
FROM python:3.10-slim


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


WORKDIR /app


# Copy requirements first for caching
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt


# Copy project
COPY . /app


# Ensure non-root
RUN useradd -m appuser || true
USER appuser


ENV PORT 5000
EXPOSE 5000


# default command: start both api and dashboard in fake/demo mode
CMD ["python", "run.py"]