import os
import json
import uuid
import base64
import asyncio
import random
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
        "'denoise_strength' (float between 0.45 for subtle silhouette/outfit edits and 0.85 for major pose/scene changes)."
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


def build_comfy_workflow(positive_prompt: str, denoise: float, input_filename: str | None) -> dict:
    seed = random.randint(1, 9999999999)
    ckpt_name = "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"

    workflow = {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": f"raw photo, 8k uhd, photorealistic, {positive_prompt}", "clip": ["4", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "cartoon, anime, blurry, deformed, bad anatomy, watermark, low quality", "clip": ["4", 1]},
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "studio_edit", "images": ["8", 0]}},
    }

    if input_filename:
        workflow["10"] = {"class_type": "LoadImage", "inputs": {"image": input_filename}}
        workflow["11"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["4", 2]}}
        latent_link = ["11", 0]
        actual_denoise = denoise
    else:
        workflow["5"] = {"class_type": "EmptyLatentImage", "inputs": {"width": 768, "height": 1024, "batch_size": 1}}
        latent_link = ["5", 0]
        actual_denoise = 1.0

    workflow["3"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": seed,
            "steps": 25,
            "cfg": 6.5,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": actual_denoise,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": latent_link,
        },
    }
    return workflow


async def run_on_kaggle(comfy_url: str, plan: dict, model_b64: str | None) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        uploaded_name = None
        if model_b64:
            img_bytes = base64.b64decode(model_b64)
            files = {"image": (f"model_{uuid.uuid4().hex[:8]}.png", img_bytes, "image/png")}
            up_res = await client.post(f"{comfy_url}/upload/image", files=files)
            up_res.raise_for_status()
            uploaded_name = up_res.json()["name"]

        workflow = build_comfy_workflow(
            positive_prompt=plan.get("diffusion_prompt", ""),
            denoise=float(plan.get("denoise_strength", 0.65)),
            input_filename=uploaded_name,
        )

        q_res = await client.post(f"{comfy_url}/prompt", json={"prompt": workflow})
        q_res.raise_for_status()
        prompt_id = q_res.json()["prompt_id"]

        for _ in range(45):
            await asyncio.sleep(2)
            hist_res = await client.get(f"{comfy_url}/history/{prompt_id}")
            hist_data = hist_res.json()
            if prompt_id in hist_data:
                outputs = hist_data[prompt_id].get("outputs", {})
                if "9" in outputs and outputs["9"]["images"]:
                    img_info = outputs["9"]["images"][0]
                    view_res = await client.get(
                        f"{comfy_url}/view",
                        params={
                            "filename": img_info["filename"],
                            "subfolder": img_info.get("subfolder", ""),
                            "type": img_info.get("type", "output"),
                        },
                    )
                    return base64.b64encode(view_res.content).decode("utf-8")

        raise HTTPException(status_code=504, detail="GPU took too long to render. Try again.")


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

    edited_b64 = await run_on_kaggle(STATE["comfy_url"], plan, req.model_image_b64)

    return {
        "status": "done",
        "edit_plan": plan,
        "sourced_outfit_image": outfit_img_url,
        "edited_image_b64": edited_b64,
            }
    
