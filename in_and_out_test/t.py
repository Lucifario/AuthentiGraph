import json
import webbrowser
import time

with open("reviews_metadata_100.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

print("Opening tabs... (Opening 10 at a time to not crash your browser)")

for i, row in enumerate(papers):
    pdf_url = row['pdf_url']
    if not pdf_url.startswith("http"):
        pdf_url = f"https://openreview.net{pdf_url}"
        
    webbrowser.open(pdf_url)
    
    # Pause every 10 tabs so you can click download and close them
    if (i + 1) % 10 == 0:
        input(f"Opened {i + 1}/100. Download these, close the tabs, and press Enter to open the next 10...")