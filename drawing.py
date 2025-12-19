def get_middle_pixel(pixel_array):
    min_x = min([pixel[0][0] for pixel in pixel_array])
    max_x = max([pixel[0][0] for pixel in pixel_array])
    min_y = min([pixel[0][1] for pixel in pixel_array])
    max_y = max([pixel[0][1] for pixel in pixel_array])

    middle = (round((min_x+max_x) / 2), round((min_y + max_y) / 2))

    return middle

def get_width(pixel_array):
    min_x = min([pixel[0][0] for pixel in pixel_array])
    max_x = max([pixel[0][0] for pixel in pixel_array])
    return max_x - min_x