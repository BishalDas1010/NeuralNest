# NeuralNest

## RAG API

Install the project dependencies and start the API from the repository root:

```bash
python -m pip install -r requirements.txt
python -m uvicorn UI.RAG.main:app --reload
```

Set `GROQ_API_KEY` in the environment or in `UI/RAG/.env` before starting the
API. For the `.env` file, add a line like:

```dotenv
GROQ_API_KEY=your_groq_api_key
```
