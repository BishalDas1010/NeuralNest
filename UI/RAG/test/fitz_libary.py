from pathlib import Path
import fitz
from pathlib import Path
import pymupdf

pdf_path = Path(__file__).resolve().parents[1] / "mml-book.pdf"
doc = fitz.open(pdf_path)

#total page
print(f"total page: {len(doc)}")


#number of page and number of image
for number, pages in enumerate(doc):
    text = pages.get_text()

    print(f"pages: {number+1}")
    print(text)





output_dir = Path("extracted_images")
output_dir.mkdir(exist_ok=True)

doc = fitz.open(pdf_path)

for page_num, page in enumerate(doc):
    images = page.get_images(full=True)

    for image_num, image_info in enumerate(images):
        xref = image_info[0]

        image_data = doc.extract_image(xref)

        image_bytes = image_data["image"]
        extension = image_data["ext"]

        image_path = (
            output_dir
            / f"page_{page_num + 1}_image_{image_num + 1}.{extension}"
        )

        image_path.write_bytes(image_bytes)

        print(f"Saved: {image_path}")

doc.close()