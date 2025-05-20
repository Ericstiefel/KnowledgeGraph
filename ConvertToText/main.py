from pathlib import Path
from ConvertToText.pdf import extract_text_and_images_from_pdf
from ConvertToText.ppt import extract_from_pptx
from ConvertToText.pt_model import getCaptionFromImage

def getData(input_path):
    path = Path(input_path)
    ext = path.suffix.lower()

    if ext != '.pdf' and ext != '.pptx':
        raise ValueError(f"Unsupported file type: {ext}")
    
    ordered = []

    if ext == ".pdf":
        ordered = extract_text_and_images_from_pdf(input_path)
    else:
        ordered = extract_from_pptx(input_path)

    concatonated = ""

    for type, datapoint in ordered:
        if type == 'image':
            try:
                caption = getCaptionFromImage(datapoint).strip()
                datapoint = caption + '.' if caption else ''
            except Exception as e:
                print(f"⚠️ Error generating caption for {datapoint}: {e}")
                datapoint = ''
        concatonated += datapoint + ' '

    return concatonated.strip()
