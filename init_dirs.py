import os
import json

# Base directory - now relative to the project directory
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Directories to create
DIRS = [
    "kb",
    "vector_store",
    "chat_vector_store"
]

# Files to create with initial content
FILES = {
    "subjects.json": {"subjects": ["Default Subject"]},
    "chat_history.json": {"exchanges": []},
    "phrases.json": {
        "greetings": ["Hello", "Hi there"],
        "farewells": ["Goodbye", "See you later"],
        "acknowledgments": ["I understand", "Got it"]
    },
    "settings.json": {
        "voice": {
            "apiKey": "",
            "voiceId": "21m00Tcm4TlvDq8ikWAM"
        },
        "llm": {
            "openai": {
                "apiKey": "",
                "model": "gpt-4-turbo-preview"
            },
            "groq": {
                "apiKey": "",
                "model": "mixtral-8x7b-32768"
            },
            "perplexity": {
                "apiKey": "",
                "model": "sonar-medium-online"
            }
        },
        "recorder": {
            "silenceThreshold": 500,
            "silenceDuration": 5.0,
            "minRecordingDuration": 1.0,
            "maxRecordingDuration": 60.0
        },
        "system": {
            "fileLocations": {
                "kbDir": os.path.join(BASE_DIR, "kb"),
                "vectorStoreDir": os.path.join(BASE_DIR, "vector_store"),
                "chatHistoryFile": os.path.join(BASE_DIR, "chat_history.json"),
                "phrasesFile": os.path.join(BASE_DIR, "phrases.json")
            }
        }
    }
}

def main():
    # Create base directory if it doesn't exist
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
        print(f"Created base directory: {BASE_DIR}")
    
    # Create subdirectories
    for dir_name in DIRS:
        dir_path = os.path.join(BASE_DIR, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
    
    # Create files with initial content
    for filename, content in FILES.items():
        file_path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=4)
            print(f"Created file: {file_path}")

if __name__ == "__main__":
    main()
    print("\nInitialization complete! You can now run app_v1_5o_alt.py") 