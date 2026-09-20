# CalorieSnap v3 plan

## Done
- [x] v3 API: Foundry Claude with forced tool use, OpenAI route, FastVLM local route
- [x] INDB verification with logged match method
- [x] Telemetry, correction flywheel, admin analytics, CSV export
- [x] UI: Snap, Log, Admin. Demo-data fallback
- [x] End-to-end tests against stub Foundry and FastVLM servers
- [x] Compose file on port 8021 beside the v2 stack

- [x] v3.2: picker matching with alias cache, suspect-data guard, honest totals, garnish rule, audit script

- [x] v3.3: 30-candidate shortlist, same/close picker grading, suspect check on the value actually used, match tester, release comparison

- [x] v3.4: full-coverage candidates never cut, wrong_match verdict with pinned aliases, admin pin/clear, per-release consistency, live miss list

- [x] v3.5: lookalike tells and correction-driven prompt hints (CONFUSION_MIN_COUNT)

- [x] WhatsApp bot v1: photo to breakdown, corrections by reply, daily total, HMAC, dedup, allowlist, hashed user keys
- [x] v3.6: snaps tagged by channel (web, whatsapp)

## Verify on the XPS (cannot be tested off-box)
- [x] Live Foundry call with the real key (5 snaps, 0 failures, $0.0020/photo measured)
- [x] HTTPS from the container to Azure through Comcast SSL inspection
- [ ] FastVLM contract: path and field names against fastvlm_server.py
- [ ] Haiku deployment accepts `temperature: 0` with forced tool use

## Verify for WhatsApp (needs Meta account and a public URL)
- [ ] Webhook handshake through the tunnel
- [ ] Live media download and reply with a real token
- [ ] Permanent System User token before inviting pilot users

## Next
- [ ] Text and voice meal logging over WhatsApp ("2 roti, 1 katori dal")
- [ ] Move API and bot to Azure Container Apps for a stable webhook address
- [ ] Run scripts/audit_indb.py. Fix implausible energy rows at the INDB import (Gulab Jamun with khoya 586 kcal/100g, Paneer pulao 4,876 kcal/plate)
- [ ] Fix the name_local parser in the INDB import: '(toasted)' is being read as a local name
- [ ] Move CalorieSnap to its own Foundry resource. Metrics on entitymap-foundry mix both projects
- [ ] Benchmark: 50+ labeled photos across Haiku, OpenAI mini, fastvlm-local
- [ ] Turn on INDB_NAMES_IN_PROMPT, compare match rate and cost per photo
- [ ] Replace LIKE fallbacks with nomic-embed-text match at 0.72 if Admin shows them under 70%
- [ ] Image-embedding retrieval over corrected photos (after ~200 corrections)
- [ ] Curate top "Dishes missing from INDB" rows
