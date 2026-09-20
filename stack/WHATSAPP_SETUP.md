# WhatsApp bot setup

Photo in, per-dish calories out, corrections by reply. About 30 minutes.

## 1. Meta side (free test number, up to 5 recipients)

1. https://developers.facebook.com > My Apps > Create App > type Business
2. Add the **WhatsApp** product. Open **WhatsApp > API Setup**
3. Copy the **temporary access token** and the **Phone number ID**
4. Under "To", add your own phone number and confirm the code
5. **App settings > Basic**: copy the **App secret**

The temporary token lasts 24 hours. For the pilot, create a System User
in Business Settings and generate a permanent token with
`whatsapp_business_messaging` and `whatsapp_business_management`.

## 2. Configure and start

```powershell
cd D:\BHTCorp\bhtlabs-pocs\calorie-snap\stack
copy whatsapp_config.example.json data-v3\whatsapp_config.json
notepad data-v3\whatsapp_config.json
$env:CS_BASE_IMAGE = docker inspect calorie-snap-api --format '{{.Config.Image}}'
docker compose -f docker-compose.v3.yml up -d --build
curl.exe http://localhost:8022/health
```

Every `*_set` field in the health output should be `true`.

## 3. Give Meta a public HTTPS address

Meta must reach the webhook from the internet. Your machine has no inbound
route, so use a tunnel for the pilot:

```powershell
winget install Cloudflare.cloudflared
cloudflared tunnel --url http://localhost:8022
```

It prints an address like `https://something.trycloudflare.com`. If
Comcast SSL inspection blocks cloudflared, use ngrok instead
(`ngrok http 8022`). The quick-tunnel address changes on every restart.
For anything beyond a demo, move both containers to Azure Container Apps
and point Meta at that address.

## 4. Subscribe the webhook

**WhatsApp > Configuration > Webhook > Edit**

- Callback URL: `https://<tunnel-address>/webhook`
- Verify token: the `WA_VERIFY_TOKEN` you invented
- Click Verify and save, then **Manage** and subscribe to `messages`

## 5. Test

Send "hi" from your phone to the test number, then a food photo.

```powershell
docker logs calorie-snap-whatsapp --tail 30
```

WhatsApp snaps appear in Admin tagged `whatsapp`, and corrections sent by
reply show up in the same tables as web corrections.

## Rules that matter

- The bot only replies to messages users send first. That is free-form
  and allowed for 24 hours after each user message. Messaging anyone
  first requires a Meta-approved template.
- Signature checking is on whenever `WA_APP_SECRET` is set. Never run a
  public webhook without it.
- Phone numbers are not stored. Users are keyed by a salted hash.
- Going beyond the 5-number test sandbox requires a verified Meta
  Business account and your own number.
