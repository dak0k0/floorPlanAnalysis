import cv2

def get_pixels_by_room(image):
    # accepts opencv image object
    # returns a dictionary of contour -> num pixels in contour

    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def count_pixels_in_contour(contour):
        min_x = min([contour[i][0][0] for i in range(len(contour))])
        max_x = max([contour[i][0][0] for i in range(len(contour))])
        min_y = min([contour[i][0][1] for i in range(len(contour))])
        max_y = max([contour[i][0][1] for i in range(len(contour))])
        
        num_pixels = 0

        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                if cv2.pointPolygonTest(contour, (x, y), False) and image[y, x] == 0:
                    num_pixels += 1

        return num_pixels

    num_contours = len(contours)
    
    result = {}

    for contour_num in range(num_contours):
        num_pixels = count_pixels_in_contour(contours[contour_num])
        result[contour_num] = num_pixels

    return result, contours
    

