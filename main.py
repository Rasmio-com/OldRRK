import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from tqdm import tqdm

# Constants
KSIZE = 3
RHO = 1.0
THRESHOLD = 26
MIN_LINE_LENGTH = 191
MIN_HORIZONTAL_LINE_LENGTH = 140
MAX_LINE_GAP = 1
HORIZONTAL_DILATATION = 5
VERTICAL_DILATATION = 9
MAX_EXPANSION_DISTANCE = 50

# File paths
INPUT_DIR = "./input"
BOXES_DIR = "./boxes"
RESULT_DIR = "./result"

"""
Image Processing Algorithm Documentation

This script processes newspaper images to detect and highlight individual advertisement boxes.
The algorithm follows these main steps:

1. Image Loading and Preprocessing
2. Edge Detection
3. Line Separation and Enhancement
4. Line Detection
5. Line Removal
6. Text Thickening
7. Contour Filling
8. Bounding Box Creation
9. Final Bounding Box Filtering and Drawing

Each step is crucial in isolating and identifying individual advertisement boxes within the newspaper layout.
"""

def sort_newspaper_boxes(filtered_rects):
    """
    Sort bounding boxes on a newspaper page from right to left and top to bottom within each column.
    Dynamically determines the number of columns based on mean column width.
    
    Args:
    filtered_rects: List of tuples (x, y, w, h, area, aspect_ratio)
    
    Returns:
    Sorted list of bounding boxes
    """
    if not filtered_rects:
        return []

    # Calculate the mean width of the bounding boxes
    mean_width = np.mean([rect[2] for rect in filtered_rects])
    
    # Estimate the image width as the maximum x-coordinate plus the width of that box
    image_width = max(rect[0] + rect[2] for rect in filtered_rects)
    
    # Estimate the number of columns
    num_columns = max(3, min(int(image_width / mean_width), len(filtered_rects)))
    
    # Group boxes into columns
    x_coordinates = np.array([[rect[0]] for rect in filtered_rects])
    
    if len(x_coordinates) < num_columns:
        return None
    
    kmeans = KMeans(n_clusters=num_columns, random_state=0).fit(x_coordinates)
    
    # Create a dictionary to store boxes for each column
    columns = {i: [] for i in range(num_columns)}
    
    for rect, label in zip(filtered_rects, kmeans.labels_):
        columns[label].append(rect)
    
    # Calculate the average x-coordinate of the column's boxes' centers
    column_centers = {
        label: np.mean([rect[0] + rect[2] / 2 for rect in boxes])
        for label, boxes in columns.items()
    }
    
    # Sort columns based on their center x-coordinate from right to left
    sorted_columns_labels = sorted(column_centers, key=column_centers.get, reverse=True)
    
    # Sort within columns and concatenate
    sorted_rects = []
    for label in sorted_columns_labels:
        # Sort boxes in this column from top to bottom
        columns[label].sort(key=lambda r: r[1])
        sorted_rects.extend(columns[label])
    
    return sorted_rects

def load_and_preprocess_image(image_path):
    """
    Load the image and convert it to grayscale.
    
    Args:
    image_path: Path to the input image
    
    Returns:
    Tuple of (original image, grayscale image)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load the image. Please check the file path.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

def detect_edges(gray):
    """
    Apply Sobel edge detection to identify horizontal and vertical edges.
    
    Args:
    gray: Grayscale image
    
    Returns:
    Tuple of (vertical edge image, horizontal edge image)
    """
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=KSIZE)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=KSIZE)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    return abs_grad_x, abs_grad_y

def enhance_lines(abs_grad_x, abs_grad_y):
    """
    Enhance horizontal and vertical lines using thresholding and morphological operations.
    
    Args:
    abs_grad_x: Vertical edge image
    abs_grad_y: Horizontal edge image
    
    Returns:
    Tuple of (enhanced vertical lines, enhanced horizontal lines)
    """
    vertical_lines_thresh = cv2.threshold(abs_grad_x, 0, 255, cv2.THRESH_OTSU)[1]
    horizontal_lines_thresh = cv2.threshold(abs_grad_y, 0, 255, cv2.THRESH_OTSU)[1]

    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    vertical_lines_morph_open = cv2.morphologyEx(vertical_lines_thresh, cv2.MORPH_OPEN, kernel_vertical)
    horizontal_lines_morph_open = cv2.morphologyEx(horizontal_lines_thresh, cv2.MORPH_OPEN, kernel_horizontal)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
    vertical_lines_morph_open = cv2.morphologyEx(vertical_lines_morph_open, cv2.MORPH_DILATE, kernel_dilate)
    horizontal_lines_morph_open = cv2.morphologyEx(horizontal_lines_morph_open, cv2.MORPH_DILATE, kernel_dilate)

    blur_thresh_x = cv2.GaussianBlur(vertical_lines_morph_open, (KSIZE, KSIZE), 0)
    blur_thresh_y = cv2.GaussianBlur(horizontal_lines_morph_open, (KSIZE, KSIZE), 0)

    return blur_thresh_x, blur_thresh_y

def detect_lines(blur_thresh_x, blur_thresh_y):
    """
    Detect and expand horizontal and vertical lines using Hough Line Transform.
    
    Args:
    blur_thresh_x: Enhanced vertical lines image
    blur_thresh_y: Enhanced horizontal lines image
    
    Returns:
    Combined line image
    """
    def expand_horizontal_line(y, x1, x2):
        left = x1
        right = x2
        while left > 0 and vertical_line_image[y, left] == 0:
            left -= 1
        while right < vertical_line_image.shape[1] - 1 and vertical_line_image[y, right] == 0:
            right += 1
        return left, right

    def expand_vertical_line(x, y1, y2):
        top = y1
        bottom = y2
        while top > 0 and horizontal_line_image[top, x] == 0:
            top -= 1
        while bottom < horizontal_line_image.shape[0] - 1 and horizontal_line_image[bottom, x] == 0:
            bottom += 1
        return top, bottom
    
    horizontal_line_image = np.zeros_like(blur_thresh_y)
    vertical_line_image = np.zeros_like(blur_thresh_x)

    vertical_lines = cv2.HoughLinesP(blur_thresh_x, RHO, np.pi / 180, THRESHOLD, np.array([]), MIN_LINE_LENGTH, MAX_LINE_GAP)
    if vertical_lines is not None:
        for line in vertical_lines:
            for x1, y1, x2, y2 in line:
                x = (x1 + x2) // 2
                top, bottom = expand_vertical_line(x, min(y1, y2), max(y1, y2))
                cv2.line(vertical_line_image, (x, top), (x, bottom), 255, 10)

    horizontal_lines = cv2.HoughLinesP(blur_thresh_y, RHO, np.pi / 180, THRESHOLD, np.array([]), MIN_HORIZONTAL_LINE_LENGTH, 2)
    if horizontal_lines is not None:
        for line in horizontal_lines:
            for x1, y1, x2, y2 in line:
                y = (y1 + y2) // 2
                left, right = expand_horizontal_line(y, min(x1, x2), max(x1, x2))
                cv2.line(horizontal_line_image, (left, y), (right, y), 255, 5)
    
    line_image = cv2.bitwise_or(horizontal_line_image, vertical_line_image)
    kernel = np.ones((3, 3), np.uint8)
    line_image = cv2.dilate(line_image, kernel, iterations=1)

    return line_image

def remove_lines(gray, horizontal_lines_morph_open, vertical_lines_morph_open, line_image):
    """
    Remove detected lines from the original image.
    
    Args:
    gray: Grayscale image
    horizontal_lines_morph_open: Enhanced horizontal lines
    vertical_lines_morph_open: Enhanced vertical lines
    line_image: Combined line image
    
    Returns:
    Image with lines removed
    """
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    clean_thresh = cv2.subtract(thresh, horizontal_lines_morph_open)
    clean_thresh = cv2.subtract(clean_thresh, vertical_lines_morph_open)
    clean_thresh = cv2.subtract(clean_thresh, line_image)
    return clean_thresh

def thicken_text(clean_thresh):
    """
    Apply morphological dilation to thicken the text.
    
    Args:
    clean_thresh: Image with lines removed
    
    Returns:
    Image with thickened text
    """
    dilatation_type = cv2.MORPH_RECT
    element = cv2.getStructuringElement(dilatation_type,
                                        (2 * HORIZONTAL_DILATATION + 1, 2 * VERTICAL_DILATATION + 1),
                                        (HORIZONTAL_DILATATION, VERTICAL_DILATATION))
    dilatation_thresh = cv2.dilate(clean_thresh, element)
    return dilatation_thresh

def fill_contours(dilatation_thresh):
    """
    Fill the contours of the thickened text.
    
    Args:
    dilatation_thresh: Image with thickened text
    
    Returns:
    Image with filled contours
    """
    filled_thresh = dilatation_thresh.copy()
    contours, _ = cv2.findContours(dilatation_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        cv2.drawContours(filled_thresh, [cnt], -1, 255, cv2.FILLED)
    return filled_thresh

def create_bounding_boxes(filled_thresh, line_image):
    """
    Create initial bounding boxes around the filled regions.
    
    Args:
    filled_thresh: Image with filled contours
    line_image: Combined line image
    
    Returns:
    Image with initial bounding boxes
    """
    bounding_box1 = filled_thresh.copy()
    contours, _ = cv2.findContours(bounding_box1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(bounding_box1, (x, y), (x + w, y + h), 255, cv2.FILLED)
    bounding_box1 = cv2.subtract(bounding_box1, line_image)
    return bounding_box1

def filter_and_draw_boxes(image, bounding_box1, save_boxes, boxes_path):
    """
    Filter bounding boxes and draw final boxes on the original image.
    
    Args:
    image: Original image
    bounding_box1: Image with initial bounding boxes
    save_boxes: Boolean indicating whether to save individual box images
    boxes_path: Path to save individual box images
    
    Returns:
    Image with final bounding boxes drawn
    """
    image_out = image.copy()
    contours, _ = cv2.findContours(bounding_box1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bounding_rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / h if h != 0 else 0
        bounding_rects.append((x, y, w, h, area, aspect_ratio))

    image_area = image_out.shape[1] * image_out.shape[0]

    filtered_rects = []
    for rect in bounding_rects:
        x, y, w, h, area, aspect_ratio = rect
        
        if area < 0.005 * image_area or area > 0.30 * image_area:
            continue
        
        if aspect_ratio < 0.1 or aspect_ratio > 10:
            continue

        if w / image_out.shape[1] * 100 < 10 or w / image_out.shape[1] * 100 > 50:
            continue
        
        overlap = False
        for f_rect in filtered_rects:
            f_x, f_y, f_w, f_h = f_rect[:4]
            if (x < f_x + f_w and x + w > f_x and y < f_y + f_h and y + h > f_y):
                overlap_area = (min(x+w, f_x+f_w) - max(x, f_x)) * (min(y+h, f_y+f_h) - max(y, f_y))
                if overlap_area / min(area, f_rect[4]) > 0.7:
                    overlap = True
                    break
        
        if not overlap:
            filtered_rects.append(rect)

    if len(filtered_rects) == 0:
        return None
    
    sorted_rects = sort_newspaper_boxes(filtered_rects)
    if not sorted_rects:
        return None

    if save_boxes:
        os.makedirs(boxes_path, exist_ok=True)

    for i, (x, y, w, h, _, _) in enumerate(sorted_rects):
        cv2.rectangle(image_out, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image_out, str(i+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        if save_boxes:
            expand_ratio = 0.06
            ex = max(0, int(x - w * expand_ratio))
            ey = max(0, int(y - h * expand_ratio))
            ew = min(image.shape[1] - ex, int(w * (1 + 2 * expand_ratio)))
            eh = min(image.shape[0] - ey, int(h * (1 + 2 * expand_ratio)))

            box_image = image[ey:ey+eh, ex:ex+ew]
            box_filename = os.path.join(boxes_path, f"box_{i+1}.jpg")
            cv2.imwrite(box_filename, box_image)

    return image_out

def process_image(image_path, save_steps=False, save_boxes=True, boxes_path=None):
    """
    Process a single newspaper image to detect advertisement boxes.
    
    Args:
    image_path: Path to the input image
    save_steps: Boolean indicating whether to save intermediate steps
    save_boxes: Boolean indicating whether to save individual box images
    boxes_path: Path to save individual box images
    
    Returns:
    Processed image with bounding boxes drawn
    """
    image, gray = load_and_preprocess_image(image_path)
    if save_steps:
        cv2.imwrite("1-grayscale.png", gray)

    abs_grad_x, abs_grad_y = detect_edges(gray)
    if save_steps:
        cv2.imwrite("2-abs_grad_x.png", abs_grad_x)
        cv2.imwrite("3-abs_grad_y.png", abs_grad_y)

    blur_thresh_x, blur_thresh_y = enhance_lines(abs_grad_x, abs_grad_y)
    if save_steps:
        cv2.imwrite("4-vertical_lines_morph_open.png", blur_thresh_x)
        cv2.imwrite("5-horizontal_lines_morph_open.png", blur_thresh_y)

    line_image = detect_lines(blur_thresh_x, blur_thresh_y)
    if save_steps:
        cv2.imwrite("6-line_image.png", line_image)

    clean_thresh = remove_lines(gray, blur_thresh_y, blur_thresh_x, line_image)
    if save_steps:
        cv2.imwrite("7-clean_thresh.png", clean_thresh)

    dilatation_thresh = thicken_text(clean_thresh)
    if save_steps:
        cv2.imwrite("8-dilatation_thresh.png", dilatation_thresh)

    filled_thresh = fill_contours(dilatation_thresh)
    if save_steps:
        cv2.imwrite("9-filled_thresh.png", filled_thresh)

    bounding_box1 = create_bounding_boxes(filled_thresh, line_image)
    if save_steps:
        cv2.imwrite("10-bounding_box1.png", bounding_box1)

    image_out = filter_and_draw_boxes(image, bounding_box1, save_boxes, boxes_path)
    if save_steps and image_out is not None:
        cv2.imwrite("11-final_output.png", image_out)

    return image_out

def process_newspapers(start_year, end_year):
    """
    Process newspaper images for a range of years.
    
    Args:
    start_year: Starting year for processing
    end_year: Ending year for processing
    """
    for year in range(start_year, end_year - 1, -1):
        for current_folder, _, files in os.walk(os.path.join(INPUT_DIR, str(year))):
            image_files = [f for f in files if f.lower().endswith('.jpg')]
            for image_file in tqdm(image_files, desc=f"Processing {year}"):
                image_path = os.path.join(current_folder, image_file)
                boxes_path = current_folder.replace(INPUT_DIR, BOXES_DIR)
                boxes_path = os.path.join(boxes_path, image_file).replace(".jpg","")
                image_out = process_image(image_path, save_steps=False, save_boxes=True, boxes_path=boxes_path)
                if image_out is None:
                    continue

                result_folder = current_folder.replace(INPUT_DIR, RESULT_DIR)
                os.makedirs(result_folder, exist_ok=True)
                result_path = os.path.join(result_folder, f"box_{image_file}")
                cv2.imwrite(result_path, image_out)

if __name__ == "__main__":
    START_YEAR = 1381
    END_YEAR = 1307
    process_newspapers(START_YEAR, END_YEAR)