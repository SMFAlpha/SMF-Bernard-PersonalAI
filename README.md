# Bernard Personal AI Assistant

Bernard is a sophisticated AI assistant that combines advanced language models with voice interaction capabilities. It features a modern web interface, real-time streaming responses, and context-aware conversations powered by vector similarity search.

## Features

- **Multi-Modal Interaction**
  - Text chat interface with real-time streaming responses
  - Voice input with speech-to-text capabilities
  - Text-to-speech output for Bernard's responses
  - Support for multiple voice models and configurations

- **Advanced Language Model Integration**
  - Support for OpenAI's GPT-4 and GPT-3.5 models
  - Integration with Groq's high-performance inference API
  - Configurable model parameters (temperature, max tokens, etc.)

- **Context-Aware Conversations**
  - Knowledge base integration with support for multiple file formats (PDF, DOCX, TXT)
  - Vector similarity search for relevant context retrieval
  - Conversation history tracking with subject categorization
  - Configurable context retrieval parameters

- **Modern Web Interface**
  - Clean, responsive design using Tailwind CSS
  - Real-time updates with Socket.IO
  - Settings management interface
  - Subject/topic organization

## Prerequisites

- Node.js 18.x or later
- API keys for:
  - OpenAI (for language model and embeddings)
  - ElevenLabs (for voice synthesis and recognition)
  - Groq (optional, for alternative language model)
  - Perplexity (optional, for additional AI capabilities)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bernard-ai.git
   cd bernard-ai
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create a `.env` file in the root directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   ELEVENLABS_API_KEY=your_elevenlabs_key
   GROQ_API_KEY=your_groq_key
   PERPLEXITY_API_KEY=your_perplexity_key
   ```

4. Create necessary directories:
   ```bash
   mkdir -p static/temp_audio data/kb data/vector_store/kb_store data/vector_store/chat_store
   ```

## Usage

1. Start the server:
   ```bash
   npm start
   ```

2. Open your browser and navigate to `http://localhost:3000`

3. Configure your settings:
   - Click the "Settings" button to configure language model and voice settings
   - Adjust context retrieval parameters as needed

4. Start interacting:
   - Type messages in the text input
   - Use the recording buttons for voice input
   - Create and switch between subjects for organized conversations

## Directory Structure

```
.
├── server.js              # Main server file
├── static/               # Static files
│   ├── index.html       # Web interface
│   └── temp_audio/      # Temporary audio files
├── data/                # Data storage
│   ├── kb/             # Knowledge base files
│   ├── vector_store/   # Vector embeddings
│   ├── subjects.json   # Subject list
│   ├── settings.json   # User settings
│   └── chat_history.json # Conversation history
└── package.json        # Project dependencies
```

## API Endpoints

- `POST /upload` - Upload knowledge base files
- `POST /upload_audio` - Upload audio recordings
- `GET /settings` - Retrieve current settings
- `POST /settings` - Update settings
- `GET /subjects` - Get available subjects
- `POST /subjects` - Update subjects list

## WebSocket Events

- `start_session` - Initialize a new chat session
- `get_completion` - Get AI response
- `transcribe` - Convert audio to text
- `speak` - Convert text to speech
- `stream_response` - Receive streaming AI response
- `audio_ready` - Audio file ready for playback

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 