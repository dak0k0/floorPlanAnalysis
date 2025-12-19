import cv2
import numpy as np

def px_sqft_ratio(image_path):
    img = cv2.imread(image_path)
    rows, cols = img.shape[:2]
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', round(cols/rows*900), 900)
    points = []

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('image', img)

            if len(points) == 2:
                cv2.line(img, points[0], points[1], (255, 0, 0), 2)
                cv2.imshow('image', img)

    
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)

    print("Click two ends of any dimension line, then press any key")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) == 2:
        # Calculate pixel distance
        pixel_distance = np.sqrt((points[1][0] - points[0][0])**2 + 
                                (points[1][1] - points[0][1])**2)
        
        # Ask user for the real-world distance
        real_distance_ft = float(input("Enter the real-world distance in feet: "))
        
        # Calculate pixels per foot
        pixels_per_foot = pixel_distance / real_distance_ft
        pixels_per_sqft = pixels_per_foot ** 2
        
        return pixels_per_sqft, pixels_per_foot
    
    return None, None