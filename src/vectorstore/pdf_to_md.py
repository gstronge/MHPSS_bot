# Change the filename variable to the name of the PDF you want to convert
# Run this script from the root directory of the project with: uv run src/rag/pdf_to_md.py
filename = "Iasc Guidelines on Mental Health and Psychosocial Support in Emergency Settings"


import base64
import os
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

def encode_pdf(pdf_path):
    """Encode the pdf to base64."""
    try:
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {pdf_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

pdf_path = f"data/{filename}.pdf"

base64_pdf = encode_pdf(pdf_path)

api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{base64_pdf}" 
    },
    include_image_base64=True
)


# Create output directories
os.makedirs(f"data/unprocessed_md/images - {filename}", exist_ok=True)
with open(f"data/unprocessed_md/PDF - {filename}.md", "w", encoding="utf-8") as md_out:
    for idx, page in enumerate(ocr_response.pages):
        md_out.write(f"<!-- Page {idx + 1} -->\n\n")
        md_out.write(page.markdown + "\n\n---\n\n")

        for img in page.images:
            img_b64 = img.image_base64
            if img_b64:
                _, b64data = (img_b64.split(",", 1)
                              if "," in img_b64 else ("", img_b64))
                ext = "png" if img_b64.startswith("data:image/png") else "jpeg"
                img_name = f"{img.id}.{ext}"
                img_path = os.path.join("data/unprocessed_md/images", img_name)
                with open(img_path, "wb") as f:
                    f.write(base64.b64decode(b64data))


