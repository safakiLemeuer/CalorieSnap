# CalorieSnap — BHTLabs

Photo your food → get calorie estimate → fully local, no cloud.

## Architecture
iPhone 17 Pro Max → Expo React Native → FastAPI on XPS → Ollama moondream

## Step 1: Pull Moondream on your XPS

```powershell
ollama pull moondream
```

~1.5GB download. Verify it works:
```powershell
ollama run moondream "describe this model"
```

## Step 2: Start the backend

```powershell
cd D:\BHTCorp\calorie-snap\backend
docker compose up --build -d
```

Verify at http://localhost:8020/health

## Step 3: Find your XPS IP address

```powershell
ipconfig
```

Look for "IPv4 Address" under your WiFi adapter.
Example: 192.168.1.105

## Step 4: Set up Expo on Windows

```powershell
cd D:\BHTCorp\calorie-snap\mobile
npm install
npx expo start
```

A QR code appears in the terminal.

## Step 5: Install Expo Go on iPhone

Download "Expo Go" from App Store on your iPhone 17 Pro Max.
Make sure iPhone is on same WiFi as XPS.
Scan the QR code with your iPhone camera.
App loads instantly — no build needed.

## Step 6: Configure the backend URL

In the app → tap ⚙️ Settings
Enter: http://YOUR_XPS_IP:8020
Tap "Save & Test"
Green dot = connected

## Step 7: Snap your first meal

- Point camera at food
- Optional: type a hint ("large portion", "Indian thali")
- Tap Snap 📸
- Wait 10-30 seconds (Moondream inference)
- See calorie breakdown

## Accuracy notes

- Simple dishes (apple, rice bowl): ±15-25% accuracy
- Complex dishes (biryani, pizza): ±30-40% accuracy
- Use hints to improve accuracy — "large plate" or "restaurant portion"
- Estimates only — not for medical use

## Port
Backend: 8020
Container: calorie-snap-api

## Model
moondream (~1.5GB) — 1.8B parameters, fits RTX 3080 easily
Inference time: 10-30 seconds per image
