# AI Digital Twin V2

Create personalized talking avatars with AI-powered voice cloning and facial animation. This project combines multiple AI technologies to generate realistic talking avatars from a single portrait image, voice sample, and text input.

## Features

- **AI-Powered Conversations** - Generate intelligent responses using TMPT AI or use custom text input
- **Advanced Voice Cloning** - Clone any voice using FishSpeech (Chatterbox) with sentence-level batching
- **Facial Animation** - Animate portrait images with SadTalker for realistic lip-sync and expressions
- **Modern Web Interface** - Clean, responsive Gradio interface with real-time progress tracking
- **Progressive Output** - Watch results generate step-by-step: AI response → cloned voice → animated video
- **Generation History** - Track and review your previous digital twin creations
- **Dockerized Services** - Complete containerized setup with GPU acceleration support
- **Robust Pipeline** - Handles long texts with intelligent sentence splitting and audio concatenation

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
├── app.py                 # Main Gradio application (838 lines)
│                         # - Pipeline orchestration
│                         # - TMPT API integration
│                         # - Voice cloning coordination
│                         # - SadTalker animation calls
│                         # - Progressive UI updates
├── docker-compose.yml     # Service orchestration with GPU support
├── chatterbox/           # FishSpeech voice cloning service
│   └── Dockerfile        # - Containerized voice cloning API
│                         # - Runs on port 8080
├── sadtalker/            # SadTalker facial animation service
│   ├── Dockerfile        # - Containerized animation service
│   └── server.py         # - Model checking and service management
│                         # - Runs on port 7861
├── data/                 # Temporary data directory (volume mounted)
│                         # - Input files processing
│                         # - Auto-cleanup after processing
├── results/              # Generated video outputs
├── training_audio/       # Audio training samples
├── test.py               # Voice cloning testing script
└── .env                  # API keys configuration (create this)
```

## Architecture & Pipeline

### Core Services
- **Chatterbox/FishSpeech** (Port 8080) - Voice cloning API service with GPU acceleration
- **SadTalker** (Port 7861) - Facial animation service with model validation
- **Main App** (app.py) - Gradio web interface with pipeline orchestration

### Processing Pipeline
1. **Input Processing**: User provides text, voice sample, portrait image, and driving video
2. **AI Response Generation** (Optional): TMPT API generates intelligent responses
3. **Text Processing**: Long texts are split into sentences with token limits (~150 tokens)
4. **Voice Cloning**: Each sentence is processed through FishSpeech with batching
5. **Audio Concatenation**: Individual sentence audio files are combined using FFmpeg
6. **Facial Animation**: SadTalker animates the portrait using the cloned voice and driving video
7. **Progressive Output**: Results are streamed to the UI as each stage completes

### Technical Features
- **Sentence-level Batching**: Handles long texts by processing sentences individually
- **GPU Memory Management**: Optimized resource allocation across services
- **Volume Mounting**: Shared data directories between containers and host
- **Error Handling**: Robust error handling with fallback mechanisms
- **Generation History**: Persistent tracking of user generations with metadata

## Troubleshooting

- **GPU Issues**: Ensure NVIDIA Docker runtime is installed and GPU is accessible
- **Model Download Fails**: Verify your Hugging Face token has correct permissions
- **Port Conflicts**: Check if ports 7861 or 8080 are already in use
- **API Errors**: Verify your TMPT API key is valid and has sufficient credits

## GPU Requirements

- **Memory**: 12GB+ VRAM recommended
- **CUDA**: Compatible with CUDA 12.x
- **Driver**: Latest NVIDIA drivers with Docker GPU support
- **Docker**: NVIDIA Container Toolkit for GPU access in containers

## Development & Testing

### Testing Voice Cloning
Use the included `test.py` script to test voice cloning functionality:
```bash
python test.py
```

### Service Health Checks
- **SadTalker**: Check model availability and service status
- **Chatterbox**: Verify FishSpeech API responsiveness
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