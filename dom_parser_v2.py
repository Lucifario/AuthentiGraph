import os
import json
import uuid
import requests
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
GROBID_URL = "http://localhost:8070/api/processFulltextDocument"
PDF_FILE = "QDER Document Entity Representations.pdf"
OUTPUT_JSON = "base_graph_v2.json"


def process_pdf_with_grobid(pdf_path):
    """Sends the PDF to the local GROBID server and returns TEI-XML."""

    # FIX 1: Check GROBID is reachable before sending. Gives a clear error
    # instead of a cryptic ConnectionRefusedError.
    try:
        health = requests.get("http://localhost:8070/api/isalive", timeout=3)
        if health.status_code != 200:
            raise RuntimeError("GROBID server is not alive. Run: docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to GROBID. Is it running? "
                           "Start it with: docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0")

    print(f"Sending {pdf_path} to GROBID...")
    with open(pdf_path, 'rb') as f:
        files = {'input': (os.path.basename(pdf_path), f, 'application/pdf')}
        response = requests.post(
            GROBID_URL,
            files=files,
            data={'consolidateHeader': '1'},  # Asks GROBID to enrich metadata
            timeout=60  # Large PDFs can take time
        )

    if response.status_code != 200:
        raise Exception(f"GROBID Error: {response.status_code} - {response.text[:300]}")

    return response.text


def parse_tei_xml(xml_content):
    """Parses GROBID TEI-XML into the AuthentiGraph DOM Schema."""

    # FIX 2: Use 'lxml-xml' parser explicitly, which handles TEI namespaces correctly.
    # 'xml' alias works only if lxml is installed; being explicit avoids silent fallback.
    # If lxml isn't installed: pip install lxml
    soup = BeautifulSoup(xml_content, 'lxml-xml')

    sections = []
    bibliography = []
    hetero_edges = []

    # --- 1. EXTRACT BIBLIOGRAPHY ---
    # GROBID packages bibliography in <listBibl> with IDs like "b0", "b1"
    print("Extracting Bibliography...")
    list_bibl = soup.find('listBibl')
    if list_bibl:
        for bibl in list_bibl.find_all('biblStruct'):
            # FIX 3: 'xml:id' attribute access. In lxml-parsed XML, the attribute
            # key is the full Clark notation: '{http://www.w3.org/XML/1998/namespace}id'
            # Use .get with both forms to be safe across parser versions.
            cite_id = (
                bibl.get('{http://www.w3.org/XML/1998/namespace}id')
                or bibl.get('xml:id')
            )
            raw_text = " ".join(bibl.get_text(separator=" ").split())

            if cite_id and raw_text:
                bibliography.append({
                    "cite_id": f"Citation_{cite_id}",
                    "raw_text": raw_text,
                    # None means "not yet verified" — downstream code must treat
                    # None as UNKNOWN, not as False (not phantom). Verified in a
                    # separate Semantic Scholar pass to avoid blocking PDF parsing.
                    "semantic_scholar_verified": None
                })

    # --- 2. EXTRACT SECTIONS & BLOCKS ---
    print("Extracting Sections and Paragraph Blocks...")

    # FIX 4: Don't chain .find('text').find('body') — if 'text' returns None
    # (namespace issue), the second find crashes. Find 'body' directly.
    body = soup.find('body')

    if not body:
        print("  WARNING: No <body> found in GROBID output. "
              "PDF may be scanned/image-only or parsing failed.")
        return sections, bibliography, hetero_edges

    for div in body.find_all('div', recursive=False):
        head = div.find('head')
        section_title = head.get_text(strip=True) if head else "Untitled Section"
        sec_id = f"sec_{str(uuid.uuid4())[:8]}"

        current_section = {
            "section_id": sec_id,
            "title": section_title,
            "blocks": []
        }

        for p in div.find_all('p'):
            paragraph_text = p.get_text(separator=" ", strip=True)

            if len(paragraph_text) < 20:  # Skip footnotes / stray single lines
                continue

            block_id = f"b_{str(uuid.uuid4())[:8]}"
            inline_citations = []

            for ref in p.find_all('ref', type='bibr'):
                target = ref.get('target')  # e.g. "#b0"
                if target:
                    target_id = f"Citation_{target.replace('#', '')}"
                    inline_citations.append(target_id)
                    hetero_edges.append({
                        "source": block_id,
                        "relation": "CITES_INLINE",
                        "target": target_id
                    })

            current_section["blocks"].append({
                "block_id": block_id,
                "type": "paragraph",
                "text": paragraph_text,
                "extracted_citations": inline_citations
            })

            hetero_edges.append({
                "source": sec_id,
                "relation": "CONTAINS_BLOCK",
                "target": block_id
            })

        if current_section["blocks"]:
            sections.append(current_section)

    return sections, bibliography, hetero_edges


def main():
    xml_content = process_pdf_with_grobid(PDF_FILE)
    sections, bibliography, hetero_edges = parse_tei_xml(xml_content)

    paper_graph = {
        "paper_metadata": {
            "paper_id": os.path.basename(PDF_FILE),
            "parser": "GROBID_v0.8.0"
        },
        "paper_DOM": {
            "sections": sections,
            "bibliography": bibliography
        },
        "heterogeneous_edges": hetero_edges
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(paper_graph, f, indent=4)

    print(f"\nDone. {len(sections)} sections | "
          f"{len(bibliography)} citations | "
          f"{len(hetero_edges)} edges -> {OUTPUT_JSON}")


if __name__ == "__main__":
    main()