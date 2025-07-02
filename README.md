# AI Digital Twin V2

Create personalized talking avatars with AI-powered voice cloning and facial animation.

## Features

- **AI-Powered Conversations** - Generate responses using TMPT AI or use raw text input
- **Voice Cloning** - Clone any voice using FishSpeech technology
- **Facial Animation** - Animate still images with SadTalker
- **Modern Web Interface** - Clean, dark-themed Gradio interface
- **Progressive Output** - See results as each stage completes
- **Dockerized Services** - Complete containerized setup with GPU support

## Requirements

- **Docker** and **Docker Compose**
- **NVIDIA GPU** with Docker GPU support
- **CUDA 12.8 or higher** installed locally
- **API Keys** (see installation below)

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/arnaybhatia/AI-Digital-Twin-V2.git
cd AI-Digital-Twin-V2
```

2. **Create a `.env` file** in the project root with your API keys:
```env
TMPT_API_KEY=your_tmpt_api_key_here
HUGGING_FACE_HUB_TOKEN=your_huggingface_token_here
```

3. **Start the services:**
```bash
docker compose up -d --build
```

4. **Access the application:**
   - Open your browser to: `http://localhost:3000` (or run `python app.py` locally)
   - SadTalker service: `http://localhost:7861`
   - FishSpeech API: `http://localhost:8080`

## API Keys Setup

### TMPT API Key
1. Visit [TMPT.ai](https://tmpt.ai) and create an account
2. Generate an API key from your dashboard
3. Add it to your `.env` file as `TMPT_API_KEY`

### Hugging Face Token
1. Visit [Hugging Face](https://huggingface.co/settings/tokens)
2. Create a new token with read permissions
3. Add it to your `.env` file as `HUGGING_FACE_HUB_TOKEN`

## Usage

1. **Upload your media files:**
   - Voice sample (audio file to clone)
   - Portrait image (still image to animate)
   - Driving video (for motion reference)

2. **Enter your text:**
   - Toggle "Use AI Response" ON for AI-generated responses
   - Toggle OFF to use your raw text directly

3. **Generate your digital twin:**
   - Click "Generate Digital Twin"
   - Watch as results appear progressively:
     1. AI response text
     2. Cloned voice audio
     3. Animated talking video

## Project Structure

```
AI-Digital-Twin-V2/
├── app.py                 # Main Gradio application
├── docker-compose.yml     # Service orchestration
├── fishspeech/           # FishSpeech voice cloning service
│   └── Dockerfile
├── sadtalker/            # SadTalker facial animation service
│   ├── Dockerfile
│   └── server.py
├── data/                 # Temporary data (auto-cleaned)
├── results/              # Generated videos
└── .env                  # API keys (create this)
```

## Services

- **FishSpeech** (Port 8080) - Voice cloning API service
- **SadTalker** (Port 7861) - Facial animation service  
- **Main App** - Gradio web interface (run locally)

## Troubleshooting

- **GPU Issues**: Ensure NVIDIA Docker runtime is installed and GPU is accessible
- **Model Download Fails**: Verify your Hugging Face token has correct permissions
- **Port Conflicts**: Check if ports 7861 or 8080 are already in use
- **API Errors**: Verify your TMPT API key is valid and has sufficient credits

## GPU Requirements

- **Memory**: 12GB+ VRAM recommended
- **CUDA**: Compatible with CUDA 12.x
- **Driver**: Latest NVIDIA drivers with Docker GPU support