# utils/ocr_api.py
import requests

from config.config import OCR_SPACE_API_KEY

def extract_text_ocr_space(file, is_pdf=False):
    """Call OCR.space API to extract text from image or PDF."""
    url = 'https://api.ocr.space/parse/image'
    
    payload = {
        'isOverlayRequired': False,
        'apikey': OCR_SPACE_API_KEY,
        'language': 'eng',
        'OCREngine': 2,
        'isCreateSearchablePdf': False,
        'filetype': 'pdf' if is_pdf else 'jpg',
    }

    files = {'file': file}
    response = requests.post(url, files=files, data=payload)

    try:
        result = response.json()
        if result['IsErroredOnProcessing']:
            return None, result.get('ErrorMessage', ['Unknown error'])[0]
        parsed = result['ParsedResults'][0]['ParsedText']
        return parsed, None
    except Exception as e:
        return None, str(e)
