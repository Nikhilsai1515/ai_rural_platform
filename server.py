import google.generativeai as genai
from fastapi import FastAPI

app = FastAPI()

# Configure the Google Generative AI API with your API key
genai.configure(api_key="AIzaSyD3ixt-Zq59NZ6XVuIoIQQDN5FPK_ACvfc")  # Replace with your actual API key

@app.get("/list-models/")
async def list_models():
    """
    Endpoint to list available models. Use this to debug and identify supported models.
    """
    try:
        models = genai.list_models()
        return {"models": [model.name for model in models]}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

@app.get("/ask-ai/")
async def ask_ai(q: str):
    """
    Main endpoint to interact with the AI model.
    """
    try:
        if not q:
            return {"error": "Query parameter 'q' is required"}

        models = genai.list_models()
        if not models:
            return {"error": "No models found for Generative Language API"}

        model_name = models[0].name  # Dynamically select the first available model
        print(f"Using model: {model_name}")

        # Call embed_content and properly iterate through the generator
        response_generator = genai.embed_content(model=model_name, content=q)
        
        embeddings = []
        for item in response_generator:
            embeddings.append(item)

        # Validate response
        if not embeddings:
            return {"error": "No valid response received from the model"}

        return {"embedding": embeddings}  # Return embedding data

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
