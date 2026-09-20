# CalorieSnap v3

Photo of an Indian meal in, per-dish calories out. The model names dishes
and estimates portions. Calories come only from INDB (ICMR-NIN).

```
photo (768 px JPEG, resized client-side)
  -> vision.identify()   Foundry Claude (forced tool use) | Foundry OpenAI route | FastVLM local
  -> matching.verify()   alias -> exact -> picker over token-retrieved candidates -> fallback
  -> telemetry           every snap, error, and user correction
  -> UI                  Snap, Log, Admin
```

## Layout

| File | Role |
|---|---|
| `main.py` | FastAPI app and endpoints |
| `vision.py` | Model calls, stdlib urllib only. FastVLM `Answer:` parser |
| `indb.py` | INDB lookups, candidate retrieval, data plausibility guard. The only calorie source |
| `matching.py` | alias, exact, model-picked candidate, substring fallback |
| `telemetry.py` | Event log, correction flywheel, admin analytics |
| `config.py` | Env first, then `/data/caloriesnap_config.json` |
| `static/ui.html` | Single-page UI. Falls back to demo data with no API |
| `tests/test_api.py` | End-to-end tests against a stub Foundry and FastVLM |

## Run

```powershell
cd D:\BHTCorp\bhtlabs-pocs\calorie-snap\stack
# Build on the existing image so pip is never needed:
$env:CS_BASE_IMAGE = docker inspect calorie-snap-api --format '{{.Config.Image}}'
docker compose -f docker-compose.v3.yml up -d --build
mkdir data-v3 -ErrorAction SilentlyContinue
copy caloriesnap_config.example.json data-v3\caloriesnap_config.json   # then edit
```

Open http://localhost:8021. Config edits apply on the next request. No
restart needed.

## Settings

| Key | Purpose |
|---|---|
| `FOUNDRY_ENDPOINT`, `FOUNDRY_API_KEY` | Cloud mode |
| `FOUNDRY_MODEL` | Default deployment name |
| `FOUNDRY_MODELS` | Extra deployments, comma separated, for benchmarking |
| `FASTVLM_URL` | Enables `fastvlm-local`. Default when no cloud key is set |
| `FASTVLM_IMAGE_FIELD` | JSON field for the base64 image. Default `image_b64` |
| `FASTVLM_TEXT_FIELD` | Reply field with the text. Auto-detected when unset |
| `INDB_NAMES_IN_PROMPT` | `1` appends all INDB names to the prompt, cached |
| `PICKER_MODEL` | Claude deployment for name matching. Defaults to `FOUNDRY_MODEL` |
| `CONFUSION_MIN_COUNT` | Corrections needed before a confusion enters the prompt. Default 1. Raise for public use |
| `MODEL_PRICES` | JSON, USD per million tokens as `[input, output]` |

## FastVLM contract (the one assumption)

`vision._identify_local` POSTs `{"image_b64": "..."}` as JSON to
`FASTVLM_URL` and reads text from the reply. If `fastvlm_server.py` uses a
different path, field name, or multipart upload, change the three
`FASTVLM_*` settings or the 10 lines in `_identify_local`. Nothing in the
FastVLM container changes.

## Test

```
python3 tests/test_api.py
```

## Audit the INDB data

```powershell
Get-Content scripts\audit_indb.py -Raw | docker exec -i calorie-snap-api-v3 python3 > indb_audit.csv
```

## WhatsApp bot

`calorie-snap-whatsapp/` is a thin webhook over the v3 API. See `WHATSAPP_SETUP.md`.
Test: `python3 calorie-snap-whatsapp/tests/test_bot.py` (starts the real API, stubs Foundry and Meta).
