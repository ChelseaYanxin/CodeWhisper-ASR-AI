# StudyAI Assistant Documentation

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
CodeWhisper is a Flask-based web application designed to help students and educators with various study-related tasks. The application integrates multiple AI technologies including speech recognition, text processing, and intelligent chatbot capabilities.

## Features
- **Multiple Study Tools**:
  - Code Whisper
  - Notes Helper
  - Slide to Note
  - Speech to Note
  - Additional Resources

## Project Structure
```
project/
├── app.py
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

## Technical Details

### Dependencies
- Flask
- PyMuPDF (fitz)
- SpeechBrain
- PyTorch
- Transformers
- PyGithub
- rake-nltk
- OpenAI
- Torchaudio

### Key Components

#### ASR (Automatic Speech Recognition)
- Uses Wav2Vec2 model from Facebook/Meta
- Supports multiple audio formats (wav, mp3, flac, ogg, mp4, m4a)
- Implements custom SpeechBrain brain class for audio processing

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

## Installation

1. Clone the repository
2. Install required dependencies:
```bash
pip install flask pymupdf speechbrain torch torchaudio transformers PyGithub rake-nltk openai
```
3. Set up environment variables
4. Create an `uploads` directory in the project root

## Environment Variables
```bash
GITHUB_TOKEN=your_github_token
OPENAI_API_KEY=your_openai_api_key
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Access the application at `http://localhost:5000`

3. Use different tools:
   - Upload PDFs for text extraction
   - Record or upload audio for transcription
   - Submit text for keyword extraction and GitHub repository search
   - Interact with the AI teaching assistant

## Notes
- The application runs in debug mode by default
- Ensure sufficient disk space for uploaded files
- Regularly clean the uploads directory
- Monitor API usage limits for GitHub and OpenAI services