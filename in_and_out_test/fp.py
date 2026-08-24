import os
import requests
import time
from tqdm import tqdm

os.makedirs("parsed_xml", exist_ok=True)
pdf_files = [f for f in os.listdir("pdfs") if f.endswith(".pdf")]

print(f"Parsing {len(pdf_files)} PDFs using LOCAL GROBID...")

# Pointing to your local Docker container instead of Hugging Face
url = "http://localhost:8070/api/processFulltextDocument"

for pdf_file in tqdm(pdf_files):
    pdf_path = os.path.join("pdfs", pdf_file)
    xml_path = os.path.join("parsed_xml", pdf_file.replace(".pdf", ".tei.xml"))

    if os.path.exists(xml_path):
        continue

    try:
        with open(pdf_path, 'rb') as f:
            files = {'input': (pdf_file, f, 'application/pdf')}
            # Fast timeout since it is running on your own machine
            res = requests.post(url, files=files, timeout=30) 
            
            if res.status_code == 200:
                with open(xml_path, 'w', encoding='utf-8') as out:
                    out.write(res.text)
            else:
                print(f"\nFailed to parse {pdf_file}: Status {res.status_code}")
    except Exception as e:
        print(f"\nError connecting to local GROBID for {pdf_file}: {e}")

print("🎉 Local Parsing Complete!")