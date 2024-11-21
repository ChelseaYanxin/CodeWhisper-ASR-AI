# CodeWhisper :mage:

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [API Routes](#api-routes)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Usage](#usage)

## Overview
CodeWhisper is a Flask-based web application designed to help students and educators with various study-related tasks. The application integrates multiple technologies including ASR, OCR, and intelligent chatbot capabilities.

## Features
- **PDF Text Extraction**: Convert PDF documents(including images) into clean, formatted text.
- **Speech-to-Text**: Convert audio recordings into text transcriptions
- **Keyword Search**: Extract keywords from text and search related GitHub repositories
- **AI Teaching Assistant**: Interactive chatbot for computer science education
- **Multiple Study Tools**:
  - Code Whisper
  - Notes Helper
  - Slide to Note
  - Speech to Note
  - Additional Resources

## Project Structure
```
project/
├── main.py
├── templates/
│   ├── CodeWhisper.html
│   ├── NotesHelper.html
│   ├── SlideToNote.html
│   ├── SpeechToNote.html
│   ├── AdditionalResources.html
│   ├── TeachingAssistant.html
│   └── results.html
└── uploads/
```

## Technical Details :writing_hand:

### Dependencies :toolbox:
- Flask
- PyMuPDF (fitz)
- SpeechBrain
- PyTorch
- Transformers
- PyGithub
- rake-nltk
- OpenAI
- Torchaudio
- pytesseract
- pdf2image
- opencv-python
- numpy
- Pillow
- librosa
- speechbrain

### Key Components 🔎

#### Audio Preprocessing 
- Loads and resamples audio files:
  - Converts audio to 16kHz sampling rate for consistent processing.
- Automatic silence removal:
  - Trims unnecessary silence or noise at the beginning and end of audio.
	- Normalizes audio:
	-	Adjusts volume levels for uniform loudness across files.
	-	Handles audio formats:
	-	Supports common formats like WAV, MP3, and FLAC.
	-	Converts stereo to mono:
	-	Ensures compatibility with single-channel processing models.
- Converts raw audio to features:
	-	Uses Wav2Vec2Processor to convert audio signals into input tensors.
	-	Handles varying audio lengths:
	-	Pads or truncates sequences to match model input requirements.
	-	Noise reduction:
	-	Reduces background noise for better recognition accuracy.
	-	Dynamic range compression:
	-	Ensures uniform audio dynamics for model stability.
- Speech-to-Text (STT) Inference:
	-	Model-driven transcription:
	-	Utilizes Wav2Vec2ForCTC for automatic speech recognition.
	-	Batch processing support:
	-	Handles multiple languages:


#### PDF Processing with OCR :notebook:
- Extracts text from both digital and scanned PDF documents
- Implements intelligent OCR detection and processing
- Supports multiple languages recognition
- Image preprocessing for better OCR accuracy:
  - Automatic image enhancement
  - Noise reduction
  - Contrast optimization
  - Sharpening
- Handles hybrid PDFs (mix of digital and scanned content)
- Cleans and formats extracted text
- Handles formatting for titles, subtitles, and bullet points
- Removes excessive whitespace and line breaks


#### PDF Processing
- Extracts text from PDF documents
- Cleans and formats text content
- Handles formatting for titles, subtitles, and bullet points
- Removes excessive whitespace and line breaks

#### GitHub Integration
- Searches repositories based on extracted keywords
- Ranks results by stars
- Returns top 5 most relevant repositories

#### AI Chat Integration
- Uses OpenAI's GPT-4 model
- Implements computer science teaching assistant functionality
- Provides contextualized responses to student queries

## API Routes

### Main Pages
- `/` - Main landing page
- `/NotesHelper` - Notes assistance tool
- `/SlideToNote` - Slide conversion tool
- `/SpeechToNote` - Speech-to-text tool
- `/AdditionalResources` - Additional learning resources
- `/TeachingAssistant` - AI teaching assistant interface

### Processing Endpoints
#### PDF Processing
```http
POST /process_pdf
Content-Type: multipart/form-data
```
- Accepts PDF files
- Returns cleaned and formatted text

#### Audio Processing
```http
POST /process_audio
Content-Type: multipart/form-data
```
- Accepts multiple audio formats
- Returns transcribed text

#### Keyword Search
```http
POST /keyword_search
Content-Type: application/x-www-form-urlencoded
```
- Accepts text input
- Returns keywords and related GitHub repositories

#### Chat Interface
```http
POST /chat
Content-Type: application/json
```
- Accepts user messages
- Returns AI assistant responses with timestamps



## Installation :wrench:

1. Clone the repository

2. Install Python dependencies:
```bash
pip install flask pymupdf speechbrain torch torchaudio transformers PyGithub rake-nltk openai pytesseract pdf2image opencv-python numpy Pillow
```

3. Install Tesseract OCR Engine:

For Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
# Optional: Install additional language packs
sudo apt-get install tesseract-ocr-chi-sim  # Simplified Chinese
sudo apt-get install tesseract-ocr-chi-tra  # Traditional Chinese
```

For MacOS:
```bash
brew install tesseract
# Optional: Install language packs
brew install tesseract-lang
```

For Windows:
- Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
- Install to default location (Usually `C:\Program Files\Tesseract-OCR`)
- Add to system PATH

4. Install Poppler (required for pdf2image):

For Ubuntu/Debian:
```bash
sudo apt-get install poppler-utils
```

For MacOS:
```bash
brew install poppler
```

For Windows:
- Download from: http://blog.alivate.com.au/poppler-windows/
- Extract to a suitable location
- Add bin directory to system PATH

5. Set up environment variables
6. Create an `uploads` directory in the project root

## Environment Variables :game_die:
```bash
GITHUB_TOKEN=your_github_token
OPENAI_API_KEY=your_openai_api_key
# For Windows users only:
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

## Usage :star:
1. Add these codes into main.py
   ```python
   GITHUB_TOKEN = os.getenv('GITHUB_TOKEN', 'your_github_token') #please replace 'your_github_token' with your GitHub token
   chatbot_bp = Blueprint('chatbot', __name__, template_folder='templates')
   client = OpenAI(api_key='your_api_key')  # please replace 'your_api_key' with your OpenAI API key
   
   # For Windows users, uncomment and update the following line:
   # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

2. Start the Flask server:
```bash
python app.py
```

3. Access the application at `http://localhost:5000`

4. Use different tools:
   - Upload PDFs for text extraction (now supports scanned documents)
   - Record or upload audio for transcription
   - Submit text for keyword extraction and GitHub repository search
   - Interact with the AI teaching assistant

## Notes :key:
- The application runs in debug mode by default
- Ensure sufficient disk space for uploaded files
- Regularly clean the uploads directory
- Monitor API usage limits for GitHub and OpenAI services
- OCR processing may take longer for large scanned documents
- For optimal OCR results:
  - Ensure good quality scans
  - Use appropriate language packs
  - Consider preprocessing settings for specific document types
