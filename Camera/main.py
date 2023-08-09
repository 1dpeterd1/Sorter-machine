# Libraries
import cv2
import numpy

# Include camera
num_cam = int(input("Which camera do you want to use? Write number: "))
cam = cv2.VideoCapture(num_cam)

# Variables
width = 0 # Width of image
height = 0 # Height of image
thresh = 100 # Threshold for cropping

# Function of image preprocessing
def img_preprocessing(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert from BGR to GRAY(Black and White)
    gauss_image = cv2.GaussianBlur(gray_image, (7, 7), 0) # Image with gauss blur
    median_image = cv2.medianBlur(gauss_image, 3) # Image with median filter
    return gray_image

# Function of making contours
def img_contours(image, image_pp):
    ret, thresh_image = cv2.threshold(image_pp, thresh, 255, cv2.THRESH_BINARY) # Image cropped threshold
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Contour search
    contours_image = numpy.zeros(image.shape) # Make new empty image
    cv2.drawContours(contours_image, contours, -1, (0, 0, 255), 3) # Drawing contours in image
    return contours_image

#Function of making contrast
def img_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8)) # Contrast Limited Adaptive Histogram Equalization
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) # Convert from BGR to LAB color space
    l, a, b = cv2.split(lab) # Split on 3 different channels
    l2 = clahe.apply(l) # Apply CLAHE to the L-channel
    lab = cv2.merge((l2, a, b)) # Merge channels
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) # Convert from LAB to BGR
    return img

# Function of angle search
def angle_search(image):
    operatedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert from BGR to GRAY(Black and White)
    operatedImage = numpy.float32(operatedImage) # Data type change
    dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07) # Angle search
    dest = cv2.dilate(dest, None) # Result marked through extended angles
    image[dest > 0.01 * dest.max()] = [0, 0, 255] # Mark the angles in the image
    return image
    
# Main function
def main():
    while True:
        # Treatment camera
        fl, image = cam.read() # Thhe camera reading
        if not fl:  # If thre are problems with camera
            print("Problem with camera!!!")
            cont = str(input("Do you want to continu working? Write yes/no: "))
            if cont == "no":
                break
        (height, width) = image.shape[:2] # Get size of the image
        
        # The image preprocessing
        img_pp = img_preprocessing(image) # The image after preprocessing
        
        # Making contours
        img_c = img_contours(image, img_pp) # The image after searching for contours
        
        #Mark the angles in the image
        img_angle = angle_search(image)
        
        # The image showing
        cv2.imshow('Camera Angle Search', img_angle)
        cv2.imshow('Camera Contours', img_c)
        
        # Stop working
        if cv2.waitKey(30) & 0xFF == ord('s'):
            print("Stop working!")
            cv2.destroyAllWindows()
            break
    return
# Start working main function
main()

# Shutdown the camera
cam.release()
