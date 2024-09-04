import os
import shutil
import pytesseract
import concurrent.futures
import fitz  # PyMuPDF
import io
from PIL import Image
import time
from bs4 import BeautifulSoup
from trafilatura import fetch_url, extract

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

DEFAULT_TIME_LIMIT = 60  # Set a default time limit in seconds

def timed_extract_text_from_pdf(pdf_path, image_output_dir, time_limit=DEFAULT_TIME_LIMIT, page_specifier=None):
    extracted_images = []  # List to hold paths of extracted images

    def process_page(page_number, page):
        all_text = ""
        # Extract text from the page
        text = page.get_text()
        all_text += f"Page {page_number + 1} Text:\n{text}\n"

        # Extract images from the page
        images = page.get_images(full=True)
        if not images:
            return all_text

        # Process each image found on the page
        for img_index, img_info in enumerate(images):
            image_index = img_info[0]
            base_image = pdf_document.extract_image(image_index)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))

            # Convert image to RGB mode if it's in CMYK
            if image.mode == 'CMYK':
                image = image.convert('RGB')

            # Save the image
            image_filename = os.path.join(image_output_dir, f'page_{page_number + 1}_image_{img_index + 1}.png')
            image.save(image_filename)  # Save image as PNG
            print(f"Saved image: {image_filename}")
            extracted_images.append(image_filename)  # Add image path to the list

            # Use OCR for images
            image_text = pytesseract.image_to_string(image, config='--oem 1')
            all_text += f"Page {page_number + 1}, Image {img_index + 1} Text:\n{image_text}\n"

        return all_text

    start_time = time.time()
    pdf_document = fitz.open(pdf_path)
    all_text = ""

    if page_specifier is not None:
        if isinstance(page_specifier, int):
            pages_to_process = [page_specifier - 1]
        elif isinstance(page_specifier, range):
            pages_to_process = page_specifier
        else:
            raise ValueError("Invalid page specifier. Use an integer for a single page or a range for multiple pages.")
    else:
        pages_to_process = range(pdf_document.page_count)

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_page, page_number, pdf_document[page_number]) for page_number in pages_to_process]
            concurrent.futures.wait(futures)

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    all_text += result

                if time.time() - start_time > time_limit:
                    break
    finally:
        pdf_document.close()

    if not all_text.strip():  # Check if any text was extracted
        print("No text extracted from the PDF file.")
        return None, extracted_images  # Return None for text and the images list

    output_dir = os.path.join(os.getcwd(), 'extracted_text')
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'extracted_text.txt')

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(all_text)

    return output_file_path, extracted_images  # Return the text path and images list

def extract_text_from_html(html_file_path):
    # Extract text from the HTML file using BeautifulSoup
    with open(html_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    soup = BeautifulSoup(content, 'html.parser')
    extracted_text = soup.get_text()

    if not extracted_text.strip():
        print("No text extracted from the HTML file.")
        return None, []  # Return None and an empty list for images

    output_file_path = os.path.join(os.getcwd(), 'extracted_text', 'extracted_html_text.txt')

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(extracted_text)

    return output_file_path, []  # Return the file path and an empty list for images

def extract_text_from_url(url):
    downloaded = fetch_url(url)
    if downloaded is None:
        print("Failed to fetch URL.")
        return None, []
    
    extracted_text = extract(downloaded)
    if not extracted_text.strip():
        print("No text extracted from the URL.")
        return None, []
    
    output_file_path = os.path.join(os.getcwd(), 'extracted_text', 'extracted_url_text.txt')
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(extracted_text)

    return output_file_path, []  # Return the file path and an empty list for images

def process_file(file_path, image_output_dir):
    if file_path.endswith(".pdf"):
        return timed_extract_text_from_pdf(file_path, image_output_dir)
    elif file_path.endswith(".html"):
        return extract_text_from_html(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a PDF or HTML file.")

def main():
    print("Text Extraction from PDF, HTML, and URL")
    file_path_or_url = input("Enter the file path (PDF or HTML) or a URL: ")

    image_output_dir = os.path.join(os.getcwd(), 'images')
    
    # Remove existing images folder if it exists
    if os.path.exists(image_output_dir):
        shutil.rmtree(image_output_dir)
        print(f"Removed existing images directory: {image_output_dir}")

    # Create the image output directory
    os.makedirs(image_output_dir, exist_ok=True)

    if os.path.isfile(file_path_or_url):
        extracted_text_file_path, extracted_images = process_file(file_path_or_url, image_output_dir)
        if extracted_text_file_path:
            print(f"Extracted text stored in: {extracted_text_file_path}")

            # Display the extracted text
            with open(extracted_text_file_path, 'r', encoding='utf-8') as f:
                extracted_text = f.read()
                print("Extracted Text:")
                print(extracted_text)

            # Display the extracted images
            if extracted_images:
                print("Extracted Images:")
                for image_path in extracted_images:
                    print(image_path)
        else:
            print("No text extracted.")
    elif file_path_or_url.startswith("http"):
        extracted_text_file_path, _ = extract_text_from_url(file_path_or_url)
        if extracted_text_file_path:
            print(f"Extracted text stored in: {extracted_text_file_path}")

            # Display the extracted text
            with open(extracted_text_file_path, 'r', encoding='utf-8') as f:
                extracted_text = f.read()
                print("Extracted Text:")
                print(extracted_text)
        else:
            print("No text extracted from the URL.")
    else:
        print("Error: Unsupported input. Please provide a valid file path or URL.")

if __name__ == "__main__":
    main()
