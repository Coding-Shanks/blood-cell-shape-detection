import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load the microscopic blood cell image
image = cv2.imread('D.jpg')
if image is None:
    print("Error: Could not load image. Please check the file path.")
    exit()

# Convert the image to grayscale and apply Gaussian blur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Apply adaptive thresholding to separate cells from the background
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Find contours of the detected cells
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours and classify shapes
output = image.copy()
for contour in contours:
    # Approximate the contour to a polygon
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Classify the shape based on the number of vertices
    if len(approx) == 3:
        shape = "Triangle"
    elif len(approx) == 4:
        shape = "Rectangle"
    elif len(approx) > 4:
        # Check if the shape is circular or elliptical
        (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(contour)
        eccentricity = np.sqrt(1 - (minor_axis ** 2 / major_axis ** 2))
        if eccentricity < 0.5:
            shape = "Circle"
        else:
            shape = "Ellipse"
    else:
        shape = "Irregular"

    # Draw the contour and label the shape
    cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(output, shape, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Display the results
print("Thresholded Image:")
cv2_imshow(thresh)

print("Detected Blood Cells with Shapes:")
cv2_imshow(output)

# Apply adaptive thresholding to separate cells from the background
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Find contours of the detected cells
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours and classify shapes
output = image.copy()
for contour in contours:
    # Approximate the contour to a polygon
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Classify the shape based on the number of vertices
    if len(approx) == 3:
        shape = "Triangle"
    elif len(approx) == 4:
        shape = "Rectangle"
    elif len(approx) > 4:
        # Check if the shape is circular or elliptical
        (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(contour)
        eccentricity = np.sqrt(1 - (minor_axis ** 2 / major_axis ** 2))
        if eccentricity < 0.5:
            shape = "Circle"
        else:
            shape = "Ellipse"
    else:
        shape = "Irregular"

    # Draw the contour and label the shape
    cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(output, shape, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Display the results
print("Thresholded Image:")
cv2_imshow(thresh)

print("Detected Blood Cells with Shapes:")
cv2_imshow(output)
