
import logging
import os

import httpx
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from streamlit.errors import StreamlitSecretNotFoundError

# Load environment variables from the project root
load_dotenv()

MODEL_NAME = "gemini-3.5-flash-lite"

client = None

logger = logging.getLogger(__name__)


def generate_response(prompt: str) -> str:
    """
    Generate a response using the Gemini API.
    Returns an error message if the API request fails.
    """

    global client

    try:
        if client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                try:
                    api_key = st.secrets.get("GEMINI_API_KEY")
                except StreamlitSecretNotFoundError:
                    api_key = None

            if not api_key:
                return (
                    "Gemini API key is missing. Set GEMINI_API_KEY in Streamlit "
                    "Secrets for the hosted app, or in .env for local use."
                )
            client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        if response.text:
            return response.text.strip()

        return "Gemini returned an empty response."

    except (httpx.TimeoutException, TimeoutError) as error:
        logger.warning("Gemini request timed out (%s).", type(error).__name__)
        return "Gemini request timed out. Please try again."

    except genai_errors.APIError as error:
        if error.code == 429:
            logger.warning("Gemini rate limit reached (HTTP 429).")
            return "Gemini rate limit reached. Please wait a moment and try again."

        logger.warning("Gemini API request failed (HTTP %s).", error.code)
        return f"Gemini could not complete the request (HTTP {error.code}). Please try again later."

    except Exception as error:
        logger.warning("Gemini request failed (%s).", type(error).__name__)
        return "Unable to generate a response right now. Please try again."
