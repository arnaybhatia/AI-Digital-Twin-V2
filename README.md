# AI Digital Twin V2

Create personalized talking avatars with AI-powered voice cloning and facial animation. This project combines multiple AI technologies to generate realistic talking avatars from a single portrait image, voice sample, and text input.

## Features

- **AI-Powered Conversations** - Generate intelligent responses using TMPT AI or use custom text input
- **Advanced Voice Cloning** - Clone any voice using Chatterbox TTS with sentence-level batching and lazy loading
- **Facial Animation** - Animate portrait images with SadTalker for realistic lip-sync and expressions
- **Modern Web Interface** - Clean, responsive Gradio interface with real-time progress tracking
- **Progressive Output** - Watch results generate step-by-step: AI response → cloned voice → animated video
- **Generation History** - Track and review your previous digital twin creations
- **Dockerized Services** - Complete containerized setup with GPU acceleration support
- **Robust Pipeline** - Handles long texts with intelligent sentence splitting and audio concatenation
- **Memory Optimization** - Lazy loading prevents VRAM usage when services are idle

## Requirements

- **Docker** and **Docker Compose**
- **NVIDIA GPU** with Docker GPU support
- **CUDA 11.3+ or 12.x** installed locally (supports both versions)
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
CHATTERBOX_API_URL=http://localhost:8080
```

3. **Start the services:**
```bash
docker compose up -d --build
```

4. **Access the application:**
   - Run the main app: `python app.py` (opens at `http://127.0.0.1:7860`)
   - SadTalker service: `http://localhost:7861`
   - Chatterbox TTS API: `http://localhost:8080`

## API Keys Setup

### TMPT API Key
1. Visit [TMPT.ai](https://tmpt.ai) and create an account
2. Generate an API key from your dashboard
3. Add it to your `.env` file as `TMPT_API_KEY`

### Chatterbox API URL (Optional)
1. Default: `http://localhost:8080`
2. Modify in `.env` file if running on different host/port
3. Used for voice cloning service communication

## Usage

1. **Upload your media files:**
   - Voice sample (audio file to clone)
   - Portrait image (still image to animate)
   - Driving video (for motion reference)

2. **Enter your text:**
   - Toggle "Ask AI JimTwin" ON for AI-generated responses
   - Toggle OFF to use your raw text directly

3. **Generate your digital twin:**
   - Click "Generate"
   - Watch as results appear progressively:
     1. AI response text
     2. Cloned voice audio
     3. Animated talking video

## Project Structure

```
AI-Digital-Twin-V2/
├── app.py                 # Main Gradio application (628 lines)
│                         # - Pipeline orchestration
│                         # - TMPT API integration
│                         # - Voice cloning coordination
│                         # - SadTalker animation calls
│                         # - Progressive UI updates
├── docker-compose.yml     # Service orchestration with GPU support
├── chatterbox/           # Chatterbox TTS voice cloning service
│   └── Dockerfile        # - Containerized voice cloning API
│                         # - Embedded Flask server
│                         # - Runs on port 8080
├── sadtalker/            # SadTalker facial animation service
│   ├── Dockerfile        # - Containerized animation service
│   └── server.py         # - Model checking and service management
│                         # - Runs on port 7861
├── data/                 # Temporary data directory (volume mounted)
│                         # - Input files processing
│                         # - Auto-cleanup after processing
├── results/              # Generated video and audio outputs
├── temp/                 # Temporary processing directory
└── .env                  # API keys configuration (create this)
```

## Architecture & Pipeline

### Core Services
- **Chatterbox TTS** (Port 8080) - Voice cloning API service with GPU acceleration and lazy loading
- **SadTalker** (Port 7861) - Facial animation service with model validation
- **Main App** (app.py) - Gradio web interface with pipeline orchestration

### Processing Pipeline
1. **Input Processing**: User provides text, voice sample, portrait image, and driving video
2. **AI Response Generation** (Optional): TMPT API generates intelligent responses
3. **Text Processing**: Long texts are split into sentences with token limits (~150 tokens)
4. **Voice Cloning**: Each sentence is processed through Chatterbox TTS with batching
5. **Audio Concatenation**: Individual sentence audio files are combined using FFmpeg
6. **Facial Animation**: SadTalker animates the portrait using the cloned voice and driving video
7. **Progressive Output**: Results are streamed to the UI as each stage completes

### Technical Features
- **Sentence-level Batching**: Handles long texts by processing sentences individually
- **GPU Memory Management**: Optimized resource allocation with lazy loading
- **Volume Mounting**: Shared data directories between containers and host
- **Error Handling**: Robust error handling with fallback mechanisms
- **Generation History**: Persistent tracking of user generations with metadata
- **Lazy Loading**: Models load only when needed to conserve VRAM

## Troubleshooting

- **GPU Issues**: Ensure NVIDIA Docker runtime is installed and GPU is accessible
- **Model Download Fails**: Ensure internet connection and sufficient disk space
- **Port Conflicts**: Check if ports 7861 or 8080 are already in use
- **API Errors**: Verify your TMPT API key is valid and has sufficient credits

## GPU Requirements

- **Memory**: 12GB+ VRAM recommended
- **CUDA**: Compatible with CUDA 12.x
- **Driver**: Latest NVIDIA drivers with Docker GPU support
- **Docker**: NVIDIA Container Toolkit for GPU access in containers

## Development & Testing

### Testing Voice Cloning
Test the Chatterbox TTS API directly:
```bash
# Check service health
curl http://localhost:8080/health

# Test TTS generation
curl -X POST http://localhost:8080/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "audio_prompt_path": "/app/data/speaker.wav"}' \
  --output test_output.wav
```

### Service Health Checks
- **SadTalker**: Check model availability and service status
- **Chatterbox**: Verify TTS API responsiveness at `/health` endpoint
- **GPU Access**: Ensure containers can access GPU resources

### Local Development
Run the main application locally without Docker:
```bash
# Ensure services are running via Docker Compose
docker compose up -d

# Run the Gradio interface locally
python app.py
```

### Container Management
```bash
# View service logs
docker compose logs sadtalker
docker compose logs chatterbox

# Restart specific service
docker compose restart sadtalker

# Check GPU access in container
docker compose exec chatterbox nvidia-smi
```

## Credits & Acknowledgments

This project builds upon and integrates several excellent open-source projects:

### Core Technologies
- **[Chatterbox](https://github.com/resemble-ai/chatterbox)** - Advanced voice cloning and text-to-speech synthesis
- **[SadTalker](https://github.com/OpenTalker/SadTalker)** - Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation

### Additional Dependencies
- **[TMPT.ai](https://tmpt.ai)** - AI conversation generation
- **[Gradio](https://gradio.app)** - Web interface framework
- **[Docker](https://docker.com)** - Containerization platform

Special thanks to the developers and researchers who created these foundational technologies that make this AI Digital Twin project possible.