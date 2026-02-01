import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF


def detect_text_regions(image_path):
    """
    Detect text regions in an image and identify empty spaces between lines
    """
    # Load image using OpenCV
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use threshold to get clearer separation
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Dilate to combine nearby contours
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get text bounding boxes
    text_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out very small areas
        if w * h > 100:  # Minimum area threshold
            text_boxes.append((x, y, w, h))
    
    # Sort by y position to get lines
    text_boxes.sort(key=lambda box: box[1])
    
    return img, text_boxes


def find_empty_spaces(text_boxes, image_height):
    """
    Find empty spaces between text lines
    """
    if not text_boxes:
        return []
        
    spaces = []
    for i in range(len(text_boxes) - 1):
        current_box = text_boxes[i]
        next_box = text_boxes[i + 1]
        
        # Calculate the gap between current box and next box
        current_bottom = current_box[1] + current_box[3]
        next_top = next_box[1]
        gap = next_top - current_bottom
        
        # Only consider significant gaps (more than 10 pixels)
        if gap > 10:
            space_x = current_box[0]
            space_y = current_bottom
            space_w = max(current_box[2], next_box[2])  # Use the wider of the two boxes
            space_h = gap
            
            spaces.append((space_x, space_y, space_w, space_h))
    
    return spaces


def place_text_in_spaces(image_path, translation_texts, output_path="output_with_translation.png"):
    """
    Place translated text in the empty spaces of the image
    """
    img, text_boxes = detect_text_regions(image_path)
    height, width = img.shape[:2]
    spaces = find_empty_spaces(text_boxes, height)
    
    # Convert image to PIL format for easier text drawing
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    # Try to load a font (fallback to default if not available)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
    
    # Place translations in empty spaces
    for i, space in enumerate(spaces):
        if i < len(translation_texts):
            x, y, w, h = space
            text = translation_texts[i]
            
            # Calculate text size to center it in the space
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Center the text in the available space
            text_x = x + (w - text_width) // 2
            text_y = y + (h - text_height) // 2
            
            # Draw the translated text in a contrasting color
            # Using blue color for better visibility against various backgrounds
            draw.text((text_x, text_y), text, fill=(0, 0, 255), font=font)
    
    # Save the resulting image
    pil_img.save(output_path)
    return output_path


def process_pdf_page_for_translation(pdf_path, page_num, translation_texts, output_path="output.pdf"):
    """
    Process a PDF page and add translations in empty spaces between lines
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Render page to image
    mat = fitz.Matrix(2.0, 2.0)  # Scale for better resolution
    pix = page.get_pixmap(matrix=mat)
    image_path = f"temp_page_{page_num}.png"
    pix.save(image_path)
    
    # Process the image to add translations
    output_image_path = place_text_in_spaces(image_path, translation_texts)
    
    # Replace the page with the modified image
    img = cv2.imread(output_image_path)
    rgb_samples = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_bytes = cv2.imencode('.png', rgb_samples)[1].tobytes()
    
    # Create a new PDF with the modified page
    new_doc = fitz.open()
    new_page = new_doc.new_page(width=page.rect.width, height=page.rect.height)
    new_page.insert_image(page.rect, stream=img_bytes)
    
    new_doc.save(output_path)
    new_doc.close()
    
    return output_path


if __name__ == "__main__":
    # Example usage - This would be called with actual inputs
    print("Image processing module ready.")
    print("To use this module:")
    print("1. Call place_text_in_spaces(image_path, translation_texts, output_path)")
    print("2. Or call process_pdf_page_for_translation(pdf_path, page_num, translation_texts, output_path)")