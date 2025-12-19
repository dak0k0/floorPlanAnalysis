import cv2
import numpy as np
from get_px_sqft_ratio import px_sqft_ratio
from preprocessing import preprocess
from computation import get_pixels_by_room
image_path = '2.png'

# first, process the image to prepare it for computation
img_processed = preprocess(image_path)
rows, cols = img_processed.shape[:2]

# next, obtain # of pixels in each room
rooms_and_pixels, contours = get_pixels_by_room(img_processed)

# get pixel:sqft ratio
pixels_per_sqft, pixels_per_foot = px_sqft_ratio(image_path)

# convert pixels to sqft
rooms_and_area = {}
for room in rooms_and_pixels.keys():
    rooms_and_area[room] = round(rooms_and_pixels[room] / pixels_per_sqft, 2)

del rooms_and_area[0]

from drawing import get_middle_pixel, get_width

# linearly scale text size based on width relative to widest room
widest_room = max([get_width(contour) for contour in contours])
narrow_room = min([get_width(contour) for contour in contours])

for room, area in rooms_and_area.items():
    middle_px = get_middle_pixel(contours[room])
    room_width = get_width(contours[room])
    origin = (middle_px[0] - min(round(room_width/4), 25), middle_px[1])
    width_scaler = ( room_width - narrow_room) / ( widest_room - narrow_room )
    font_size = 0.4 + width_scaler*0.6
    cv2.putText(img_processed, str(area), origin, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = font_size, color = 255, thickness = 2)

# cv2.drawContours(img_processed, contours, 1, (127), -1)
cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('result', round(cols/rows*900), 900)
cv2.imshow('result', img_processed)
cv2.waitKey()