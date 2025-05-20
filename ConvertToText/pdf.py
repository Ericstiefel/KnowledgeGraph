import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO

def extract_text_and_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    ordered_content = []

    for page_number, page in enumerate(doc):
        blocks = []

        # Extract text blocks
        for block in page.get_text("dict")["blocks"]:
            if block["type"] == 0:  # text block
                text = "".join([line["spans"][0]["text"] for line in block["lines"]])
                y = block["bbox"][1]
                blocks.append(("text", y, text))

        # Extract images
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_pil = Image.open(BytesIO(img_bytes))

            # Determine y-coordinate if possible
            try:
                img_rects = page.get_image_rects(xref)
                y = img_rects[0].y0 if img_rects else float('inf')
            except:
                y = float('inf')

            blocks.append(("image", y, img_pil))

        # Sort blocks by Y-coordinate
        blocks.sort(key=lambda x: x[1])

        for kind, _, content in blocks:
            ordered_content.append((kind, content))

    return ordered_content
