import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
import google.generativeai as genai
from google.generativeai.types import GenerationConfig  # Import GenerationConfig

# --- Configuration ---
TARGET_MODEL_NAME = "models/chat-bison-001"  # Set your target model name
CANDIDATE_COUNT = 1  # For compatibility with older models like Bison

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Simple Generative AI API",
    description="An API to interact with Google's Generative AI models.",
    version="1.1.0",
)

# --- Google AI Configuration ---
model = None  # Global variable for the model instance

def configure_google_ai():
    global model
    try:
        # Load API key from environment variables
        api_key = os.getenv("AIzaSyD3ixt-Zq59NZ6XVuIoIQQDN5FPK_ACvfc")
        if not api_key:
            logger.error("GOOGLE_API_KEY environment variable not set.")
            raise ValueError("GOOGLE_API_KEY environment variable not set.")

        genai.configure(api_key=api_key)
        logger.info("Google Generative AI configured successfully.")

        # Retrieve available models and validate the target model
        available_models_info = {m.name: m.supported_generation_methods for m in genai.list_models()}
        target_model_found = False
        actual_model_name = None

        if TARGET_MODEL_NAME in available_models_info:
            target_model_found = True
            actual_model_name = TARGET_MODEL_NAME
        elif f"models/{TARGET_MODEL_NAME}" in available_models_info:
            target_model_found = True
            actual_model_name = f"models/{TARGET_MODEL_NAME}"
        elif TARGET_MODEL_NAME.split("/")[-1] in available_models_info:
            target_model_found = True
            actual_model_name = TARGET_MODEL_NAME.split("/")[-1]

        if not target_model_found:
            logger.error(f"Target model '{TARGET_MODEL_NAME}' not found in available models.")
            raise ValueError(f"Target model '{TARGET_MODEL_NAME}' not found.")

        if "generateContent" not in available_models_info[actual_model_name]:
            logger.error(f"Target model '{actual_model_name}' does not support content generation.")
            raise ValueError(f"Target model '{actual_model_name}' does not support content generation.")

        # Initialize the model instance
        model = genai.GenerativeModel(actual_model_name)
        logger.info(f"Successfully initialized generative model: {actual_model_name}")

    except Exception as e:
        logger.exception("Error configuring Google Generative AI:")
        raise SystemExit(f"Failed to configure Google Generative AI: {str(e)}") from e

configure_google_ai()  # Configure on application startup

# --- API Endpoints ---

@app.get("/")
async def read_root():
    return {"message": f"Welcome! Using AI model: {TARGET_MODEL_NAME}"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return PlainTextResponse("Not Found", status_code=404)

@app.get("/list-models/")
async def list_models_endpoint():
    if not genai.api_key:
        raise HTTPException(status_code=500, detail="Google AI SDK not configured.")
    try:
        models_list = genai.list_models()
        output_models = [
            {
                "name": model.name,
                "display_name": model.display_name,
                "description": model.description,
                "supported_generation_methods": model.supported_generation_methods,
            }
            for model in models_list
        ]
        return {"models": output_models}
    except Exception as e:
        logger.exception("Error listing models:")
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@app.get("/ask-ai/")
async def ask_ai(q: str):
    global model
    if not model:
        raise HTTPException(status_code=503, detail="Generative AI Model not initialized.")
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required.")

    try:
        generation_config = GenerationConfig(candidate_count=CANDIDATE_COUNT)

        try:
            response = await model.generate_content_async(q, generation_config=generation_config)
        except AttributeError:
            response = model.generate_content(q, generation_config=generation_config)

        # Extract text from candidates
        answers = []
        if hasattr(response.candidates, '__iter__'):  # Explicitly check if it's iterable
            for candidate in response.candidates:
                # Check if the candidate has content and parts
                if candidate.content and candidate.content.parts:
                    candidate_text = "".join(part.text for part in candidate.content.parts if hasattr(part, "text"))
                    answers.append(candidate_text.strip())  # Add strip() to remove leading/trailing whitespace
                elif hasattr(candidate, "text") and candidate.text:  # Older models might have text directly on candidate
                    answers.append(candidate.text.strip())
                elif candidate.finish_reason != "STOP":  # Handle cases where finish_reason indicates no output
                    logger.warning(f"Candidate finished with reason: {candidate.finish_reason}. Content: {candidate.content if hasattr(candidate, 'content') else 'N/A'}")
        else:
            logger.warning(f"response.candidates is not iterable: {response.candidates}")

        if not answers:
            raise HTTPException(status_code=500, detail="AI model did not generate any text answers.")

        return {"query": q, "answers": answers}

    except Exception as e:
        logger.exception("Error during AI interaction:")
        raise HTTPException(status_code=500, detail=f"Error during AI interaction: {str(e)}")

# --- Dockerized Configuration ---
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8080))  # Default port for containerized deployment
    uvicorn.run(app, host="0.0.0.0", port=port)
