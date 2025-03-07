# AI-Digital-Twin-V2

An interactive AI assistant that uses speech recognition, natural language processing, and text-to-speech to create a realistic digital twin experience.

## Features

- Real-time speech recognition with wake word detection ("Hey Jim" or "Hello Jim")
- Conversational AI powered by GPT-4o
- Natural text-to-speech synthesis 
- Interruption capability (you can interrupt the assistant while it's speaking)

## Requirements

- Python 3.8+
- OpenAI API key

## Dependencies

- `openai` - For API communication with GPT models
- `SpeechRecognition` - For audio processing
- `whisper` - For speech-to-text conversion
- `soundfile` - For audio file manipulation
- `playsound` - For playing audio responses
- `numpy` - For numerical operations
- `python-dotenv` - For environment variable management
- `kokoro` - For text-to-speech generation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI-Digital-Twin-V2.git
cd AI-Digital-Twin-V2
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install openai speechrecognition soundfile playsound numpy python-dotenv kokoro whisper
```

4. Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Start the AI Digital Twin:

```bash
python main.py
```

Once the system is running:
1. Say "Hey Jim" or "Hello Jim" to activate the assistant
2. Ask your question or give a command
3. Wait for Jim's response

Press Ctrl+C to exit the program.

## Troubleshooting

- If you encounter audio-related errors, make sure your microphone is properly connected and configured
- For "Access Denied" errors with temporary files, ensure your user has write permissions in the project directory
- If OpenAI API requests fail, verify your API key and internet connection