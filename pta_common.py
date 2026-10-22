"""
Shared helpers for the PTA treasurer agents: LLM provider setup, loading scans
as images for a vision model, and PDF report utilities.
"""

import argparse
import base64
import io
import logging
import os
import sys
from pathlib import Path
from typing import List

from PIL import Image
from fpdf import FPDF, XPos, YPos
from pdf2image import convert_from_path
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

PROVIDERS = ("gemini", "ollama")
GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_OLLAMA_MODEL = "qwen2.5vl:7b"
# Local vision models slow down sharply on large images; cap the longest side.
OLLAMA_MAX_IMAGE_SIDE = 1600

# Silence the google-genai AFC advisory triggered by langchain's internal calls.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)


# ==========================================
# LLM Provider
# ==========================================

class LLMSettings(BaseModel):
    provider: str = "gemini"
    model: str = GEMINI_MODEL


settings = LLMSettings()


def configure_llm(provider: str, model: str | None = None) -> None:
    """Selects the provider ("gemini" or "ollama") and model used by make_structured_llm."""
    if provider not in PROVIDERS:
        raise ValueError(f"LLM provider must be one of {PROVIDERS}, got '{provider}'.")
    default = GEMINI_MODEL if provider == "gemini" else os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    settings.provider = provider
    settings.model = model or default


def add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    """Adds --provider and --model options; defaults come from LLM_PROVIDER / OLLAMA_MODEL."""
    env_provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    ollama_model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    parser.add_argument(
        "-p", "--provider",
        choices=PROVIDERS,
        default=env_provider if env_provider in PROVIDERS else None,
        help="LLM provider: 'gemini' (needs GOOGLE_API_KEY) or 'ollama' (local). "
             "Defaults to $LLM_PROVIDER, or 'gemini'."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        help=f"Model name for the chosen provider (defaults: gemini={GEMINI_MODEL}, ollama={ollama_model})."
    )


def configure_llm_from_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Applies the options added by add_llm_arguments, reporting a bad LLM_PROVIDER as a usage error."""
    if args.provider is None:
        parser.error(f"LLM_PROVIDER must be 'gemini' or 'ollama', got '{os.environ.get('LLM_PROVIDER')}'.")
    configure_llm(args.provider, args.model)


def require_api_key() -> None:
    """Exits with an error if the configured provider needs an API key that is not set."""
    if settings.provider == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)


def describe_llm() -> str:
    return f"{settings.provider}: {settings.model}"


def make_structured_llm(schema: type[BaseModel]):
    """Returns a vision LLM bound to a Pydantic output schema for the configured provider."""
    if settings.provider == "gemini":
        return ChatGoogleGenerativeAI(model=settings.model, temperature=0.0).with_structured_output(schema)
    # Ollama constrains decoding to the JSON schema; num_ctx leaves room for image tokens.
    # Small local models skip optional fields, so every field is marked required here.
    llm = ChatOllama(model=settings.model, temperature=0.0, num_ctx=8192)
    json_schema = _require_all_fields(schema.model_json_schema())
    return llm.with_structured_output(json_schema, method="json_schema") | schema.model_validate


def _require_all_fields(node):
    """Marks every property of every object in a JSON schema as required."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node["required"] = list(node["properties"])
        for value in node.values():
            _require_all_fields(value)
    elif isinstance(node, list):
        for value in node:
            _require_all_fields(value)
    return node


# ==========================================
# File Conversion
# ==========================================

def _fit_for_model(img: Image.Image) -> Image.Image:
    """Downscales large pages for local Ollama models; Gemini gets the full resolution."""
    if settings.provider == "ollama" and max(img.size) > OLLAMA_MAX_IMAGE_SIDE:
        img = img.copy()
        img.thumbnail((OLLAMA_MAX_IMAGE_SIDE, OLLAMA_MAX_IMAGE_SIDE), Image.LANCZOS)
    return img


def load_file_as_base64_images(file_path: str) -> List[str]:
    """
    Accepts a path to a PDF or an Image file (.png, .jpeg, .jpg, .webp).
    Returns a list of base64-encoded PNG strings, one per page.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        images = convert_from_path(file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
        with Image.open(file_path) as img:
            images = [img.convert("RGB")]
    else:
        raise ValueError(f"Unsupported file format '{ext}' for file {file_path}")

    base64_images = []
    for img in images:
        buffered = io.BytesIO()
        _fit_for_model(img).save(buffered, format="PNG")
        base64_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
    return base64_images


def image_payload(prompt: str, file_path: str) -> list:
    """Builds a chat message content list: the prompt followed by every page of the file."""
    payload = [{"type": "text", "text": prompt}]
    for b64 in load_file_as_base64_images(file_path):
        payload.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })
    return payload


# ==========================================
# PDF Reports
# ==========================================

GREEN = (22, 128, 60)
RED = (190, 30, 45)
ORANGE = (200, 110, 0)
TABLE_HEADING_FILL = (230, 230, 230)

# fpdf2's built-in fonts only cover Latin-1, so map common typographic characters.
_PDF_REPLACEMENTS = {
    "—": "-", "–": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', "•": "*", "…": "...", "™": "(TM)",
}


def pdf_text(text) -> str:
    text = str(text if text is not None else "")
    for char, replacement in _PDF_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    return text.encode("latin-1", "replace").decode("latin-1")


def new_report_pdf() -> FPDF:
    pdf = FPDF(format="Letter")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    return pdf


def pdf_heading(pdf: FPDF, text: str, size: int) -> None:
    pdf.set_font("Helvetica", "B", size)
    pdf.multi_cell(0, size * 0.5, pdf_text(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)


def pdf_label_line(pdf: FPDF, label: str, value) -> None:
    """Writes one 'Label: value' line with a bold label."""
    pdf.set_font("Helvetica", "B", 10)
    pdf.write(5, f"{label}: ")
    pdf.set_font("Helvetica", size=10)
    pdf.write(5, pdf_text(value))
    pdf.ln(5)


def pdf_notes(pdf: FPDF, title: str, notes: List[str]) -> None:
    pdf_heading(pdf, title, 14)
    pdf.set_font("Helvetica", size=10)
    for note in notes:
        pdf.multi_cell(0, 5, pdf_text(f"- {note}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)
