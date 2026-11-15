import os
import json
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ddgs import DDGS
from bs4 import BeautifulSoup

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = {"comfy_url": os.getenv("COMFY_URL", "")}
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")


class RegisterGPU(BaseModel):
    comfy_url: str


class ChatEditRequest(BaseModel):
    prompt: str
    model_image_b64: str | None = None
    outfit_url: str | None = None


@app.get("/")
async def health():
    return {"status": "online", "gpu_connected": bool(STATE["comfy_url"])}


@app.post("/register-gpu")
async def register_gpu(data: RegisterGPU):
    STATE["comfy_url"] = data.comfy_url.rstrip("/")
    return {"status": "connected", "comfy_url": STATE["comfy_url"]}


def search_online_outfit(query: str) -> str | None:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=3))
            if results:
                return results[0]["image"]
    except Exception:
        return None
    return None


async def extract_image_from_page(page_url: str) -> str | None:
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=10) as client:
            resp = await client.get(page_url, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(resp.text, "html.parser")
            og_img = soup.find("meta", property="og:image")
            if og_img and og_img.get("content"):
                return og_img["content"]
    except Exception:
        return None
    return None


async def parse_chat_intent(user_prompt: str) -> dict:
    system_prompt = (
        "You are an AI fashion & photography studio director. Convert the user's edit request into strict JSON with keys: "
        "'action' (one of: 'outfit_swap', 'pose_edit', 'silhouette_edit', 'full_generate'), "
        "'outfit_search_query' (string or null if the user wants an outfit searched online), "
        "'diffusion_prompt' (detailed photorealistic prompt describing lighting, pose, body silhouette, and outfit), "
        "'denoise_strength' (float between 0.35 for subtle silhouette/proportion tweaks and 0.85 for full outfit/pose changes)."
    )
    async with httpx.AsyncClient(timeout=25) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openrouter/free",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            },
        )
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)


@app.post("/chat-edit")
async def chat_edit(req: ChatEditRequest):
    if not STATE["comfy_url"]:
        raise HTTPException(
            status_code=503,
            detail="Kaggle GPU is offline. Start Cell 3 in your Kaggle notebook first.",
        )

    plan = await parse_chat_intent(req.prompt)

    outfit_img_url = None
    if req.outfit_url:
        outfit_img_url = await extract_image_from_page(req.outfit_url)
    elif plan.get("outfit_search_query"):
        outfit_img_url = search_online_outfit(plan["outfit_search_query"])

    return {
        "status": "ready",
        "comfy_endpoint": f"{STATE['comfy_url']}/prompt",
        "edit_plan": plan,
        "sourced_outfit_image": outfit_img_url,
    }
