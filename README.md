# 🧠 Live Emotion Detection — Vercel + Render Deployment

> Real-time facial emotion detection using a custom-trained CNN model, deployed with React on Vercel and Flask on Render.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat&logo=flask)
![React](https://img.shields.io/badge/React-19-61DAFB?style=flat&logo=react)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat&logo=opencv)
![Vercel](https://img.shields.io/badge/Frontend-Vercel-black?style=flat&logo=vercel)
![Render](https://img.shields.io/badge/Backend-Render-46E3B7?style=flat&logo=render)

---

## 🔗 Live Demo

| | Link |
|---|---|
| 🌐 Live App | [face-emotion-detectionl.vercel.app](https://face-emotion-detectionl.vercel.app/) |
| 💻 Local Server Version | [github.com/mumer78/emotions-detection-onlocalserver](https://github.com/mumer78/emotions-detection-onlocalserver) |

> ⚠️ **Note:** The live demo may feel slow due to Render's free tier CPU limits and network latency (150–300ms per frame from Pakistan to US servers). For instant real-time performance, use the **local server version** above.

---

## ✨ Core Features

- 🎥 **Live webcam feed** — detects faces and classifies emotions in real time
- 🤖 **Custom CNN model** — trained from scratch on a facial expression dataset
- 😀 **5 Emotions detected** — Angry, Happy, Neutral, Sad, Surprise
- 📊 **Probability bars** — shows confidence percentage for each emotion live
- 🌍 **Cloud deployed** — frontend on Vercel, backend on Render
- 🧹 **Memory optimized** — lightweight inference for constrained cloud CPUs
- 📐 **Optimized frame size** — sends 240x180 compressed frames to minimize latency

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model | CNN (TensorFlow / Keras) |
| Face Detection | OpenCV Haar Cascade |
| Backend | Flask + Flask-CORS |
| Frontend | React + Vite + Tailwind CSS |
| Image Processing | NumPy, OpenCV, Base64 |
| Frontend Hosting | Vercel |
| Backend Hosting | Render |

---

## 📁 Project Structure

```
Emotion-detection/
├── emotion_model_save.keras        # Trained CNN model
├── code/
│   ├── app.py                      # Flask backend (Render optimized)
│   └── haarcascade_frontalface_default.xml
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   └── ...
    ├── package.json
    └── vite.config.js
```

---

## ⚙️ How It Works

```
Webcam (Browser on Vercel)
      ↓  captures 240x180 compressed frame
React Frontend (Vercel)
      ↓  sends base64 image via POST /api/predict
Flask Backend (Render)
      ↓  detects face with Haar Cascade (scaleFactor=1.3)
      ↓  resizes to 48x48 grayscale
      ↓  runs CNN model inference
      ↓  returns emotion + probabilities
React Frontend
      ↓  draws bounding box + emotion label
You see the result ✅
```

---

## 🚀 Run Locally (Development)

> ⚠️ You need **two terminals** running at the same time.

### 1️⃣ Clone the repo

```bash
git clone https://github.com/mumer78/Emotion-detection.git
cd Emotion-detection
```

### 2️⃣ Terminal 1 — Flask Backend

```bash
cd code
pip install flask flask-cors tensorflow opencv-python numpy
python app.py
```

You should see:
```
SUCCESS: Loaded cascade classifier
SUCCESS: Model warmed up
* Running on http://127.0.0.1:5000
```

### 3️⃣ Terminal 2 — React Frontend

```bash
cd frontend
npm install
npm run dev
```

Open: `http://localhost:5173`

---

## ☁️ Deploy Your Own

### Backend → Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set these:
   - **Root Directory:** `code`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
5. Deploy ✅

### Frontend → Vercel

1. Go to [vercel.com](https://vercel.com) → New Project
2. Connect your GitHub repo
3. Set these:
   - **Root Directory:** `frontend`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
4. Add environment variable:
   - `VITE_API_URL` = your Render backend URL
5. Deploy ✅

---

## 🧠 Model Details

- Architecture: **Custom CNN** built with TensorFlow/Keras
- Input: **48x48 grayscale** facial image
- Output: **5 emotion classes** — Angry, Happy, Neutral, Sad, Surprise
- Training: Data augmentation + Dropout regularization
- Format: `.keras`

---

## ⚡ Cloud Speed Optimizations Applied

| Optimization | Detail |
|---|---|
| Smaller frame size | Frontend captures at 240x180 instead of full resolution |
| JPEG compression | Frames compressed to ~4KB before sending |
| Faster face detection | `scaleFactor=1.3` + `minSize=(80,80)` on Haar Cascade |
| TF thread limits | `intra/inter_op_parallelism_threads = 1` for Render's shared CPU |
| Memory cleanup | `gc.collect()` after every prediction |
| Model warmup | Dummy prediction on startup to avoid cold inference delay |

---

## ❓ Troubleshooting

**App loads but no detection / very slow**
> Expected on Render free tier. 150–300ms latency per frame is normal. Use the local version for real-time speed.

**`ECONNREFUSED 127.0.0.1:5000` (local)**
> Flask backend is not running. Start Terminal 1 first.

**`Model not found`**
> Make sure `emotion_model_save.keras` is in the root folder, not inside `code/`.

**Render backend sleeping**
> Render free tier sleeps after 15 mins of inactivity. First request takes ~30 seconds to wake up.

---

## 👨‍💻 Author

**Muhammad Umer**
- 🌐 Portfolio: [portfolio-muhammad-umer.vercel.app](https://portfolio-muhammad-umer.vercel.app/)
- 💼 LinkedIn: [linkedin.com/in/muhammad-umer-247970318](https://linkedin.com/in/muhammad-umer-247970318)
- 🐙 GitHub: [github.com/mumer78](https://github.com/mumer78)

---

> 🎓 Built as part of my **4th Semester AI Course** at Forman Christian College, Lahore.
