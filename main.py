
from flask import Blueprint, Flask, render_template, request, jsonify
import fitz  # pymupdf
import re
import speechbrain as sb
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from github import Github
from rake_nltk import Rake
import os
from openai import OpenAI
from datetime import datetime
import logging
import pytesseract
from pdf2image import convert_from_path
import numpy as np
from PIL import Image
import cv2

# represent the application
app = Flask(__name__)

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN', 'your_github_token_here')
chatbot_bp = Blueprint('chatbot', __name__, template_folder='templates')
client = OpenAI(api_key='your_openai_api_key')

pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe' # your path to Tesseract executable

def preprocess_image(image):
    
   
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # transform to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # binarize the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # denoise the image
    denoised = cv2.fastNlMeansDenoising(thresh)

    # sharpen the image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    return Image.fromarray(sharpened)

def perform_ocr(image, lang='eng+chi_sim'):
    
    # process the image
    processed_image = preprocess_image(image)
    
    # convert the image to text
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed_image, lang=lang, config=custom_config)
    
    return text

def extract_clean_pdf_text(pdf_file_path):
    # open the PDF file
    doc = fitz.open(pdf_file_path)
    full_text = ""
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # extract text from the page
        text = page.get_text()
        
        # if the text is too short, perform OCR on the page
        if len(text.strip()) < 50:  # arbitrary threshold
            # convert the page to an image
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # perform OCR on the image
            text = perform_ocr(img)
        
        full_text += text + "\n"

    # clean the text
    # remove excessive line breaks and whitespace
    cleaned_text = re.sub(r'\n{2,}', '\n', full_text)
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)

    # split paragraphs
    paragraphs = cleaned_text.split("\n")

    # append a space to paragraphs that don't end with punctuation
    formatted_paragraphs = []
    for para in paragraphs:
        if para and not re.search(r'[.!?]$', para.strip()):
            formatted_paragraphs.append(para.strip() + " ")
        else:
            formatted_paragraphs.append(para.strip() + "\n")

    # add titles and bullet points
    final_paragraphs = []
    for para in formatted_paragraphs:
        if re.match(r'^[A-Z\s]+$', para.strip()):
            final_paragraphs.append(f"\n\n# {para.strip()}\n\n")
        elif re.match(r'^\d+\.\s', para.strip()):
            final_paragraphs.append(f"\n## {para.strip()}\n")
        elif re.match(r'^[-*]\s', para.strip()):
            final_paragraphs.append(f"- {para.strip()[2:]}\n")
        else:
            final_paragraphs.append(para)

    final_text = ''.join(final_paragraphs)
    return final_text

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded PDF to a temporary location
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    try:
        # extract and clean text from the PDF
        cleaned_text = extract_clean_pdf_text(file_path)
        
        # remove the temporary file
        os.remove(file_path)
        
        return jsonify({'cleaned_text': cleaned_text})
    except Exception as e:
        # remove the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': str(e)}), 500


class ASR(sb.Brain):
    def __init__(self, modules=None, hparams=None, run_opts=None):
        if modules is None:
            modules = {}

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec2 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        modules["wav2vec2"] = self.wav2vec2

        super().__init__(modules=modules, hparams=hparams, run_opts=run_opts)

    def prepare_features(self, wavs, wav_lens=None):
        if wavs.dim() == 3:
            wavs = wavs.squeeze(1)

        if hasattr(self.hparams, 'sample_rate') and self.hparams.sample_rate != 16000:
            wavs = torchaudio.transforms.Resample(
                self.hparams.sample_rate, 16000
            )(wavs)

        input_values = self.processor(
            wavs.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).input_values

        return input_values

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        input_values = self.prepare_features(wavs, wav_lens)
        input_values = input_values.to(self.device)

        outputs = self.wav2vec2(input_values)
        logits = outputs.logits

        return logits

    def compute_objectives(self, predictions, batch, stage):
        batch = batch.to(self.device)
        ids = batch.id
        tokens, token_lens = batch.tokens

        log_probs = torch.nn.functional.log_softmax(predictions, dim=-1)
        loss = self.hparams.ctc_loss(log_probs, tokens, None, token_lens)

        if stage != sb.Stage.TRAIN:
            sequence = self.decode_predictions(predictions)
            self.wer_metric.append(ids, sequence, tokens)

        return loss

    def decode_predictions(self, logits):
        predicted_ids = torch.argmax(logits, dim=-1)
        transcriptions = self.processor.batch_decode(predicted_ids)
        return transcriptions

    def transcribe_file(self, audio_file):
        waveform, sample_rate = torchaudio.load(audio_file)

        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(
                sample_rate, 16000
            )(waveform)

        input_values = self.prepare_features(waveform)
        input_values = input_values.to(self.device)

        with torch.no_grad():
            outputs = self.wav2vec2(input_values)
            logits = outputs.logits

        transcription = self.decode_predictions(logits)[0]

        return transcription

def create_asr_model():
    hparams = DotDict({
        "sample_rate": 16000,
        "ctc_loss": torch.nn.CTCLoss(),
    })

    model = ASR(
        hparams=hparams,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )

    return model


def asr_audio(speech_file_path):
    asr_model = create_asr_model()
    result = asr_model.transcribe_file(speech_file_path)
    return result





# Route for the HTML page
# 创建一个路由和视图函数的映射
@app.route('/')
def index():
    # return render_template('index.html')
    return render_template('CodeWhisper.html')

@app.route('/NotesHelper')
def NotesHelper():
    return render_template('NotesHelper.html')

@app.route('/SlideToNote')
def SlideToNote():
    return render_template('SlideToNote.html')

@app.route('/SpeechToNote')
def SpeechToNote():
    return render_template('SpeechToNote.html')

@app.route('/AdditionalResources')
def AdditionalResources():
    return render_template('AdditionalResources.html')

# Route to handle file upload and processing
@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # check the file type
    if not file.filename.endswith(('.wav', '.mp3', '.flac', '.ogg', '.mp4','.m4a')):
        return jsonify({'error': 'Unsupported file format'}), 400

    # Save the uploaded audio to a temporary location
    file_path = f'./uploads/{file.filename}'
    file.save(file_path)

    try:
        transcription = asr_audio(file_path)
    except Exception as e:
        return jsonify({'error': f'ASR processing failed: {str(e)}'}), 500

    # get the ASR result
    # transcription = asr_audio(file_path)  可能回来

    # # Extract and clean text from the PDF
    # cleaned_text = extract_clean_pdf_text(file_path)

    # Return the cleaned text as a response
    return jsonify({'cleaned_text': transcription})


@app.route('/keyword_search', methods=['GET', 'POST'])
def key_search():
    if request.method == 'POST':
        # Get text input from user
        text = request.form.get('text', '')

        # Extract keywords using RAKE algorithm
        rake = Rake()
        rake.extract_keywords_from_text(text)

        # Get the top 3 keywords, making sure to only take the phrases (not the scores)
        keyword_scores = rake.get_ranked_phrases_with_scores()[:3]
        keywords = [phrase for score, phrase in keyword_scores]  # Fixed: properly unpack score and phrase

        # Search GitHub projects
        try:
            g = Github(GITHUB_TOKEN)
            # Combine keywords into search query
            query = ' OR '.join(keywords)
            repositories = []

            # Search repositories and get top 5 results
            repos = g.search_repositories(query, sort='stars', order='desc')
            for repo in repos[:5]:
                repositories.append({
                    'name': repo.name,
                    'url': repo.html_url,
                    'description': repo.description,
                    'stars': repo.stargazers_count
                })

            return render_template('results.html',
                                   keywords=keywords,
                                   repositories=repositories,
                                   text=text)
        except Exception as e:
            error = f"An error occurred during search: {str(e)}"
            return render_template('AdditionalResources.html', error=error)

    return render_template('AdditionalResources.html')


@app.route('/TeachingAssistant')
def TeachingAssistant():
    return render_template('TeachingAssistant.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json['message']

        # format the message in the way OpenAI expects
        messages = [
            {"role": "system",
             "content": "You are a helpful Computer Science Teaching Assistant. Please respond in English."},
            {"role": "user", "content": user_message}
        ]

        # send the message to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )

        assistant_message = response.choices[0].message.content

        return jsonify({
            "response": assistant_message,
            "timestamp": datetime.now().strftime("%H:%M")
        })

    except Exception as e:
        print(f"Error: {str(e)}")  # log the error
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)