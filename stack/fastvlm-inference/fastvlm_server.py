from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64, io, logging, os, re, sys, time
from pathlib import Path
from PIL import Image
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_PATH   = Path(os.environ.get("MODEL_PATH",   "/model"))
FASTVLM_REPO = Path(os.environ.get("FASTVLM_REPO", "/fastvlm"))

PROMPT = "You are CalorieSnap. Identify this Indian food and estimate calories.\nReply exactly:\nDISH: poha\nCALORIES: 220\nCONFIDENCE: HIGH"
_DISHES = ["poha","upma","idli","dosa","dal makhani","dal tadka","chole","aloo paratha","biryani","butter chicken","paneer","rajma","samosa","rice","roti"]

app = FastAPI(title="FastVLM Inference", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model = _tokenizer = _image_processor = _device = None

class QueryRequest(BaseModel):
    image_base64: str

def _extract_dish(raw):
    m = re.search(r"DISH\s*[:\-=]\s*(.+)", raw, re.IGNORECASE)
    if m: return m.group(1).strip()
    for d in _DISHES:
        if d in raw.lower(): return d.title()
    return "Unknown dish"

def _extract_calories(raw):
    for p in [r"CALORIES\s*[:\-=]\s*(\d+)", r"(\d{2,4})\s*(?:cal|kcal|calories?)"]:
        m = re.search(p, raw, re.IGNORECASE)
        if m: return int(m.group(1))
    return 0

@app.on_event("startup")
async def startup():
    global _model, _tokenizer, _image_processor, _device
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", _device)
    sys.path.insert(0, str(FASTVLM_REPO))
    sys.path.insert(0, str(FASTVLM_REPO / "llava"))
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from transformers import CLIPImageProcessor
    disable_torch_init()
    tok, model, proc, _ = load_pretrained_model(
        model_path=str(MODEL_PATH), model_base=None,
        model_name="llava-fastvithd", device_map=_device)
    if proc is None:
        proc = CLIPImageProcessor.from_pretrained(str(MODEL_PATH), local_files_only=True)
    _tokenizer, _model, _image_processor = tok, model, proc
    _model.eval()
    log.info("FastVLM v4 ready on :8031")

@app.post("/query")
async def query(req: QueryRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    b64 = req.image_base64
    if b64.startswith("data:"): b64 = b64.split(",", 1)[1]
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")
    try:
        from llava.mm_utils import tokenizer_image_token, process_images
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        t0 = time.monotonic()

        # Build prompt exactly as predict.py does
        qs = DEFAULT_IMAGE_TOKEN + "\n" + PROMPT
        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Set pad token — this fixes bos_token_id error
        _model.generation_config.pad_token_id = _tokenizer.pad_token_id
        if _model.generation_config.bos_token_id is None:
            _model.generation_config.bos_token_id = _tokenizer.pad_token_id

        # Tokenize
        input_ids = tokenizer_image_token(
            prompt, _tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(_device)

        # Process image exactly as predict.py does
        image_tensor = process_images([img], _image_processor, _model.config)[0]

        with torch.inference_mode():
            output_ids = _model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(_device),
                image_sizes=[img.size],
                do_sample=False,
                temperature=0.0,
                num_beams=1,
                max_new_tokens=300,
                use_cache=True,
            )

        raw = _tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        elapsed = time.monotonic() - t0
        log.info("FastVLM %.1fs | %s", elapsed, raw[:150])

        dish = _extract_dish(raw)
        calories = _extract_calories(raw)
        conf_m = re.search(r"CONFIDENCE\s*[:\-=]\s*(LOW|MEDIUM|HIGH)", raw, re.IGNORECASE)
        confidence = conf_m.group(1).upper() if conf_m else ("MEDIUM" if calories > 0 else "LOW")
        return {"dish_name": dish, "total_calories": calories, "confidence": confidence,
                "elapsed_ms": int(elapsed*1000), "raw": raw, "model": "fastvlm-0.5b"}
    except Exception as e:
        log.exception("Inference error")
        raise HTTPException(500, f"Inference failed: {e}")

@app.get("/health")
async def health():
    return {"status": "ok" if _model is not None else "loading", "model_loaded": _model is not None, "device": _device}
