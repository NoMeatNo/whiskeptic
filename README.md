# whiskeptic
# Real-time Skeptic Translator

A Python application that provides real-time speech translation with a focus on detecting and translating English words in Persian speech. The application supports multiple transcription providers (OpenAI and Groq) and offers customizable post-processing options.

## Features

- Real-time speech capture and processing
- Support for multiple transcription providers:
  - OpenAI (whisper-1)
  - Groq (whisper-large-v3-turbo, whisper-large-v3)
- English word detection in Persian speech
- Translation of detected English words to Persian
- Customizable post-processing using various OpenAI models
- Session transcript saving
- Debug logging capabilities
- Configurable audio and display settings
- User-friendly GUI interface

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Groq API key (if using Groq transcription)
- Working microphone

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nomeatno/whiskeptic.git
cd whiskeptic
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your API keys:
```env
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
```

## Usage

1. Run the application:
```bash
python whiskeptic_v1.8.py
```

2. In the application:
   - Select your preferred transcription provider (OpenAI or Groq)
   - Choose the transcription model
   - Select post-processing model
   - Click "Start" to begin recording
   - Speak in Persian
   - The application will detect English words and display their translations
   - Click "Stop" to end the session

3. Settings:
   - Adjust font size and type
   - Configure chunk duration and display duration
   - Enable/disable debug logging
   - Set up transcript saving options

## Configuration

The application provides several configurable settings:

- **Audio Settings**:
  - Chunk duration: Length of audio segments to process
  - Sample rate: Audio sampling rate (default: 16000 Hz)
  - Channels: Audio channels (default: 1, mono)

- **Display Settings**:
  - Font size and type
  - Word display duration
  - Debug log visibility

- **Transcription Settings**:
  - Provider selection (OpenAI/Groq)
  - Model selection
  - Post-processing model selection

## Future Improvements

1. Support for additional languages
2. More transcription providers
3. Custom word detection rules
4. Real-time visualization of audio input
5. Enhanced error handling and recovery
6. Support for file input (pre-recorded audio)
7. Export options for translations
8. Cloud storage integration
9. User profiles and settings persistence
10. Translation memory for faster processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for Whisper and GPT models
- Groq for their transcription service
- Contributors and testers

