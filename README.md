# 🛡️ Detect The Deceptive — Multi-Modal Deepfake Detection Platform

<p align="left">
  <img src="https://img.shields.io/badge/Status-In%20Development-orange?style=flat"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi"/>
  <img src="https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white"/>
</p>

> ⚠️ **This project is actively under development.** The architecture and roadmap below describe the planned system. Features are being built and validated incrementally. Contributions and feedback welcome.

A unified GenAI verification platform designed to detect AI-synthesized images and voice clones — addressing the growing threat of deepfakes in misinformation, identity fraud, and social engineering attacks.

---

## 🎯 Problem Statement

Generative AI has made it trivially easy to fabricate realistic faces, voices, and media. Existing detection tools are siloed — separate tools for image forgery, separate tools for audio cloning. **Detect The Deceptive** aims to provide a single API and dashboard that handles both modalities, with explainable outputs that go beyond a binary "fake/real" verdict.

---

## 🏗️ Planned Architecture

```
User Upload (Image / Audio)
          │
          ▼
┌────────────────────────────────┐
│        React Frontend          │
│  Upload → Results → Visual     │
└──────────────┬─────────────────┘
               │ REST API
               ▼
┌────────────────────────────────┐
│    FastAPI Microservices       │
│  ┌────────────┐ ┌────────────┐ │
│  │ /analyze/  │ │ /analyze/  │ │
│  │   image    │ │   voice    │ │
│  └──────┬─────┘ └──────┬─────┘ │
└─────────┼──────────────┼───────┘
          │              │
          ▼              ▼
  ConvNeXt Model   Audio Classifier
  + Grad-CAM        + Feature
  Explainability      Analysis
          │              │
          ▼              ▼
     Metadata       Confidence
     Forensics        Score
          │              │
          └──────┬────────┘
                 ▼
          Unified Verdict
          + Explanation
```

---

## 🔬 Technical Approach

### Image Deepfake Detection
- **Model:** ConvNeXt-Tiny (chosen for strong performance on texture artifacts vs. ResNet/EfficientNet)
- **Training data:** FaceForensics++, DFDC datasets (planned)
- **Explainability:** Grad-CAM activation maps to highlight regions that triggered the detection
- **Metadata forensics:** EXIF analysis, DCT coefficient analysis, noise pattern inconsistency

### Voice Clone Detection
- Feature extraction: MFCC, spectral centroid, zero-crossing rate analysis
- Detection of unnatural prosody patterns and frequency artifacts common in TTS/voice conversion
- Classifier: SVM or lightweight CNN over audio spectrograms (under evaluation)

---

## 📍 Current Development Status

| Component | Status |
|---|---|
| Project architecture design | ✅ Complete |
| FastAPI backend scaffolding | 🔄 In Progress |
| ConvNeXt model training pipeline | 🔄 In Progress |
| Grad-CAM visualization | 🔄 In Progress |
| Voice analysis module | 📋 Planned |
| React frontend | 📋 Planned |
| Docker containerization | 📋 Planned |
| Integration tests | 📋 Planned |

---

## 🚀 Getting Started (Development Build)

```bash
git clone https://github.com/MeghnaaT/Detect-The-Deceptive.git
cd Detect-The-Deceptive

# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

> Full setup instructions will be updated as each module reaches a runnable state.

---

## 📁 Repository Structure *(Planned)*

```
├── backend/
│   ├── main.py                  # FastAPI entry point
│   ├── routers/
│   │   ├── image_analysis.py    # ConvNeXt inference + Grad-CAM
│   │   └── voice_analysis.py    # Audio classifier
│   ├── models/                  # Saved model weights
│   └── utils/
│       ├── metadata_forensics.py
│       └── gradcam.py
├── frontend/                    # React app (planned)
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🔮 Roadmap

- [ ] Complete ConvNeXt training pipeline with benchmark results on FaceForensics++
- [ ] Implement Grad-CAM heatmap overlay in API response
- [ ] Build and validate voice clone detection module
- [ ] Integrate both into unified FastAPI service
- [ ] Build React dashboard with upload + results visualization
- [ ] Dockerize and deploy to cloud (Hugging Face Spaces / Render)
- [ ] Publish model card with accuracy metrics and dataset details

---

## 🤝 Contributing

This project is in early development — contributions, ideas, and issue reports are especially welcome at this stage.

1. Fork the repository
2. Create a branch: `git checkout -b feat/your-contribution`
3. Open a PR or issue with a description of what you're working on

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built by [Meghna Tiwari](https://github.com/MeghnaaT) · Deepfake detection research & tooling*

