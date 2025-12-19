import cv2
import numpy as np
import math
import statistics

def preprocess(image_path):
    # accepts a png screenshot of architectural plan
    # returns a grayscale image of the walls with furniture and text erased

    image = image_path

    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    rows, cols = img.shape[:2]

    # threshold (part of binarization)
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    def in_bounds(row,col):
        return 0 <= row < rows and 0 <= col < cols

    def get_neighbors(row, col, to_edit):
        neighbors = []
        directions = [(-1, -1), (-1, 0), (-1, 1),
                    ( 0, -1),          ( 0, 1),
                    ( 1, -1), ( 1, 0), ( 1, 1)]
        for d_row, d_col in directions:
            n_row, n_col = row + d_row, col + d_col
            if in_bounds(n_row, n_col) and to_edit[n_row, n_col] == 255:
                neighbors.append((n_row, n_col))
        return neighbors

    def get_neighbors_v2(row, col, input_image):
        neighbors = []
        directions = [(-1, 0),( 0, -1),( 0, 1),( 1, 0)]
        for d_row, d_col in directions:
            n_row, n_col = row + d_row, col + d_col
            if in_bounds(n_row, n_col):
                neighbors.append((n_row, n_col))
        return neighbors

    def distance(p1, p2):
        """Manhattan distance between two pixels."""
        return abs(p1[0] - p2[0]) + abs( p1[1] - p2[1])

    def direction(p1, p2):
        d_row, d_col = p2[0] - p1[0], p2[1] - p1[1]
        length = math.hypot(d_row, d_col)
        return (d_row / length, d_col / length) if length > 0 else (0, 0)

    def angle_between(v1, v2):
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        return math.acos(np.clip(dot, -1.0, 1.0))

    def follow_line(start, to_edit, visited, guess_length):
        stack = [start]
        path = []
        visited[start] = True
        prev_dir = None
        while stack:
            r, c = stack.pop()
            path.append((r, c))
            neighbors = get_neighbors(r, c, to_edit)
            neighbors = [(nr, nc) for nr, nc in neighbors]

            if not neighbors:
                continue

            best_dir = None

            if prev_dir:
                nr, nc = int(r + prev_dir[0]), int(c + prev_dir[1])
                if (nr, nc) in neighbors:
                    next_pixel = (nr, nc)
                else:
                    next_pixel = None

            else:
                for neighbor in neighbors:
                    dir = direction((r, c), (neighbor))
                    test_scalars = [r for r in range(0,guess_length,5)]
                    are_all_white = 1
                    for scalar in test_scalars:
                        tr, tc = int(r + scalar * dir[0]), int(c + scalar * dir[1])
                        if in_bounds(tr, tc):
                            is_white = to_edit[tr, tc] == 255
                            are_all_white &= is_white
                    if are_all_white == 1:
                        best_dir = dir
                if best_dir:
                    nr, nc = int(r + best_dir[0]), int(c + best_dir[1])
                    next_pixel = (nr, nc)
                    prev_dir = best_dir
                else:
                    next_pixel = None

            if next_pixel:
                visited[next_pixel] = True
                stack.append(next_pixel)

        return path

    min_length = 20
    max_angle_variation = 0.3

    result = np.zeros_like(binary)

    def erase_small_lines(input, length):
        visited = np.zeros_like(binary, dtype=bool)

        result = input.copy()

        for row in range(rows):
            for col in range(cols):
                if result[row, col] == 255 and not visited[row, col]:
                    path = follow_line((row, col), result, visited, length)
                    if len(path) < length:
                        for(r, c) in path:
                            result[r, c] = 0
                    else:
                        for (r, c) in path:
                            result[r, c] = 255

        return result

    binary_filtered = erase_small_lines(binary, min_length)

    kernel = np.ones((8,8), np.uint8)
    small_closing = cv2.morphologyEx(binary_filtered, cv2.MORPH_CLOSE, kernel, iterations=1)

    small_closing_filtered = erase_small_lines(small_closing, 70)

    kernel = np.ones((15,15), np.uint8)
    closing = cv2.morphologyEx(small_closing_filtered, cv2.MORPH_CLOSE, kernel, iterations=1)

    kernel = np.ones((3,3), np.uint8)
    erode = cv2.morphologyEx(closing, cv2.MORPH_ERODE, kernel, iterations=2)

    from skimage.morphology import skeletonize

    skeleton = skeletonize(erode // 255).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    uniform_lines = cv2.dilate(skeleton, kernel, iterations=1)
    uniform_lines = cv2.morphologyEx(uniform_lines, cv2.MORPH_CLOSE, kernel, iterations=1)

    kernel_up = np.array([[-1, -1, -1, -1, -1], [-1, 1, 1, 1, -1]])
    kernel_right = np.array([[-1, -1], [1, -1], [1, -1], [1, -1], [-1, -1]])
    kernel_down = np.array([[-1, 1, 1, 1, -1], [-1, -1, -1, -1, -1]])
    kernel_left = np.array([[-1, -1], [-1, 1], [-1, 1], [-1, 1], [-1, -1]])

    post_kernel_up = cv2.morphologyEx(uniform_lines, cv2.MORPH_HITMISS, kernel_up)
    post_kernel_right = cv2.morphologyEx(uniform_lines, cv2.MORPH_HITMISS, kernel_right)
    post_kernel_down = cv2.morphologyEx(uniform_lines, cv2.MORPH_HITMISS, kernel_down)
    post_kernel_left = cv2.morphologyEx(uniform_lines, cv2.MORPH_HITMISS, kernel_left)

    def find_white_pixels(input_image):
        locations = []
        for row in range(rows):
            for col in range(cols):
                if input_image[row, col] == 255:
                    locations.append((row, col))

        return locations

    locations_post_kernel = post_kernel_up + post_kernel_right + post_kernel_left + post_kernel_down

    locations = find_white_pixels(locations_post_kernel)

    locations_filtered = []
    for location in locations:
        if location[0] != 0 and location[1] != 0:
            locations_filtered.append(location)

    def fix_locations(location, input_image):
        closest_white_pixel = None
        closest_dist = math.inf

        for r in range(-5,5):
            for c in range(-5,5):
                if input_image[location[0] + r, location[1] + c] == 255:
                    dist = distance((location[0] + r, location[1] + c), location)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_white_pixel = location[0] + r, location[1] + c
        return closest_white_pixel

    def closest_noncontiguous_white_pixel(input_image, coord):
        starting_points = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for direction in directions:
            dist = 0
            while input_image[coord[0] + direction[0] * dist, coord[1] + direction[1] * dist] == 255 and dist < 3:
                dist += 1
            if input_image[coord[0] + direction[0] * dist, coord[1] + direction[1] * dist] == 0:
                starting_points.append((coord[0] + direction[0] * dist, coord[1] + direction[1] * dist))

        def is_white(coordinate, image):
            return image[coordinate[0], coordinate[1]] == 255

        for starting_point in starting_points:
            length = math.sqrt((coord[0] - starting_point[0]) ** 2 +  (coord[1] - starting_point[1]) ** 2)
            case = (int((starting_point[0] - coord[0]) / length), int(((starting_point[1] - coord[1]) / length)))
            if length == 1:
                visited = set()
                    
                import queue
                q = queue.Queue()
                q.put(starting_point)

                while not q.empty():
                    curr = q.get()
                    if curr in visited:
                        continue
                    visited.add(curr)

                    # case-based neighbor filtering
                    if case == (1, 0): # positive row direction
                        neighbors = [neighbor for neighbor in get_neighbors_v2(curr[0], curr[1], input_image) if neighbor[0] > starting_point[0]]
                    elif case == (-1, 0): # negative row direction
                        neighbors = [neighbor for neighbor in get_neighbors_v2(curr[0], curr[1], input_image) if neighbor[0] < starting_point[0]]
                    elif case == (0, 1): # positive col direction
                        neighbors = [neighbor for neighbor in get_neighbors_v2(curr[0], curr[1], input_image) if neighbor[1] > starting_point[1]]
                    elif case == (0, -1):
                        neighbors = [neighbor for neighbor in get_neighbors_v2(curr[0], curr[1], input_image) if neighbor[1] < starting_point[1]]

                    for neighbor in neighbors:
                        if neighbor in visited:
                            continue
                        if not is_white(neighbor, input_image):
                            q.put(neighbor)
                        elif distance(starting_point, neighbor) > 2:
                            return (neighbor, distance(coord, neighbor))
                return (None, None)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    uniform_lines = cv2.morphologyEx(uniform_lines, cv2.MORPH_CLOSE, kernel)

    fixed_locations = []
    for location in locations_filtered:
        if uniform_lines[location[0], location[1]] == 255:
            fixed_locations.append(location)
        if uniform_lines[location[0], location[1]] == 0:
            fixed_locations.append(fix_locations(location, uniform_lines))

    locations_and_distances = {}
    for location in fixed_locations:
        if uniform_lines[location[0], location[1]] == 255:
            result = closest_noncontiguous_white_pixel(uniform_lines, location)
            if result[0] != None:
                locations_and_distances[location] = (result[0], result[1])

    distances = [ item[1] for item in locations_and_distances ]

    uniform_lines_closed = uniform_lines.copy()

    for location in fixed_locations:
        if location in locations_and_distances:
            other_point = locations_and_distances[location][0]

            dr = other_point[0] - location[0]
            dc = other_point[1] - location[1]
            
            num_pixels_drawn = 0

            if dc > 0:
                for col in range(dc):
                    uniform_lines_closed[location[0], location[1] + col] = 255
                    num_pixels_drawn += 1
            elif dc < 0:
                for col in range(0, dc, -1):
                    uniform_lines_closed[location[0], location[1] + col] = 255
                    num_pixels_drawn += 1

            if dr > 0:
                for row in range(dr):
                    uniform_lines_closed[location[0] + row, location[1]] = 255
                    num_pixels_drawn += 1
            elif dr < 0:
                for row in range(0, dr, -1):
                    uniform_lines_closed[location[0] + row, location[1]] = 255
                    num_pixels_drawn += 1

    final = cv2.dilate(uniform_lines_closed, kernel, iterations=1)

    return final