FROM python:3.12-slim

WORKDIR /app

COPY req.txt ./
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r req.txt

COPY . ./

CMD ["python", "world_cup_model.py"]
