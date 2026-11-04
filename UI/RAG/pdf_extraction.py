import hashlib
from pathlib import Path

import pymupdf


def pdf_Extraction(path, image_file_path, text__file_path):
    image_file_path = Path(image_file_path)
    text__file_path = Path(text__file_path)
    image_file_path.mkdir(parents=True, exist_ok=True)
    text__file_path.parent.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(path)
    page_count = len(doc)
    seen_hashes = set()                       # dedupe images (logos, headers)
    n_text_pages = 0
    n_images = 0

    # --- open text file once, write page by page ---
    with open(text__file_path, "w", encoding="utf-8") as text_file:

        for page_num, page in enumerate(doc):

            #  TEXT 
            page_text = page.get_text("text").strip()

            if page_text:
                text_file.write(f"\n\n========== PAGE {page_num + 1} ==========\n\n")
                text_file.write(page_text)
                n_text_pages += 1

            # ========== 2. IMAGES ==========
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]                     # image reference id

                try:
                    pix = doc.extract_image(xref)
                except Exception as e:
                    print(f"  skip image xref={xref} on page {page_num+1}: {e}")
                    continue

                data = pix["image"]               # raw bytes
                ext = pix["ext"]                  # png / jpeg / ...

                # dedupe identical images across pages
                h = hashlib.md5(data).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                # save to disk
                fname = Path(image_file_path) / f"page{page_num+1:03d}_{img_index}_{xref}.{ext}"
                fname.write_bytes(data)
                n_images += 1

    doc.close()

    # --- report ---
    print(f"✅ Text written → {text__file_path}")
    print(f"   pages with text: {n_text_pages} / {page_count}")
    print(f"✅ Images saved → {image_file_path}")
    print(f"   unique images: {n_images}")

rag_dir = Path(__file__).resolve().parent
pdf_path = rag_dir / "mml-book.pdf"


# ---- usage ----
pdf_Extraction(
    path=pdf_path,
    image_file_path=rag_dir / "extracted_images",
    text__file_path=rag_dir / "extracted_text" / "mml-book.txt",
)