# correr servicios en dev
## services_ML

cd services_ML
venv\Scripts\activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

## services llm

cd services_LLM
venv\Scripts\activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload