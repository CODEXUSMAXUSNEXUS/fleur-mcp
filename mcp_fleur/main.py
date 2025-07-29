#!/usr/bin/env python3

"""FastAPI server bridging Google Gemini function calls to Obsidian."""

from __future__ import annotations

import os
from typing import Any, Callable, Awaitable, Dict

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

OBSIDIAN_API_URL = os.getenv("OBSIDIAN_API_URL", "http://127.0.0.1:27123")
OBSIDIAN_API_KEY = os.getenv("OBSIDIAN_API_KEY")

app = FastAPI(title="Fleur MCP")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class FunctionCall(BaseModel):
    """Representation of a Gemini function call."""

    name: str
    args: Dict[str, Any] | None = None


class FunctionCallPart(BaseModel):
    function_call: FunctionCall


class CandidateContent(BaseModel):
    parts: list[FunctionCallPart]


class Candidate(BaseModel):
    content: CandidateContent


class GeminiRequest(BaseModel):
    candidates: list[Candidate]


# ---------------------------------------------------------------------------
# Obsidian interaction helpers
# ---------------------------------------------------------------------------

async def _request(
    method: str, path: str, *, content: str | None = None
) -> str:
    headers = {}
    if OBSIDIAN_API_KEY:
        headers["Authorization"] = f"Bearer {OBSIDIAN_API_KEY}"

    # S'assurer que le Content-Type est correct pour les requêtes avec corps
    if content is not None:
        headers['Content-Type'] = 'text/markdown'

    url = f"{OBSIDIAN_API_URL.rstrip('/')}/{path.lstrip('/')}"
    async with httpx.AsyncClient() as client:
        response = await client.request(method, url, content=content.encode('utf-8') if content else None, headers=headers)
        response.raise_for_status()
        return response.text


async def lire_note(chemin_fichier: str) -> str:
    """Read a note from Obsidian."""
    return await _request("GET", f"vault/{chemin_fichier}")


async def creer_ou_ecraser_note(chemin_fichier: str, contenu: str) -> str:
    """Create or replace a note in Obsidian."""
    return await _request("PUT", f"vault/{chemin_fichier}", content=contenu)


async def ajouter_a_la_note(chemin_fichier: str, contenu_a_ajouter: str) -> str:
    """Append content to an existing note in Obsidian."""
    # L'API Local REST utilise POST pour l'ajout (append)
    return await _request("POST", f"vault/{chemin_fichier}", content=contenu_a_ajouter)


TOOLS: dict[str, Callable[..., Awaitable[str]]] = {
    "lire_note": lire_note,
    "creer_ou_ecraser_note": creer_ou_ecraser_note,
    "ajouter_a_la_note": ajouter_a_la_note,
}


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

@app.post("/")
async def handle_function_call(payload: GeminiRequest) -> dict[str, Any]:
    """Handle Gemini function calls."""
    try:
        call = payload.candidates[0].content.parts[0].function_call
    except (IndexError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid request format")

    tool = TOOLS.get(call.name)
    if tool is None:
        return {
            "tool_response": {
                "name": call.name,
                "response": {"content": f"Erreur: fonction inconnue '{call.name}'"},
            }
        }

    try:
        args = call.args or {}
        result = await tool(**args)
    except Exception as exc:
        return {
            "tool_response": {
                "name": call.name,
                "response": {"content": f"Erreur: {exc}"},
            }
        }

    return {"tool_response": {"name": call.name, "response": {"content": result}}}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
