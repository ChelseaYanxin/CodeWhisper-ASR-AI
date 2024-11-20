# CodeWhisper :mage:

[Previous sections remain the same until Technical Details...]

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

### Key Components 🔎

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

[Other components remain the same...]

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