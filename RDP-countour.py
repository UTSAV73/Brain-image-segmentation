import cv2
import numpy as np
import matplotlib.pyplot as plt

def rdp(points, epsilon):
    """
    Simplify a curve using the Ramer-Douglas-Peucker algorithm.
    :param points: List of points (x, y).
    :param epsilon: Maximum distance between original curve and simplified curve.
    :return: Simplified list of points.
    """
    def perpendicular_distance(point, line_start, line_end):
        if np.array_equal(line_start, line_end):
            return np.linalg.norm(np.array(point) - np.array(line_start))
        return np.linalg.norm(np.cross(np.array(line_end) - np.array(line_start),
                                       np.array(line_start) - np.array(point))) / np.linalg.norm(np.array(line_end) - np.array(line_start))
    
    def rdp_recurse(points, start, end, epsilon, result):
        if start >= end:
            return
        max_dist = 0
        index = start
        for i in range(start + 1, end):
            dist = perpendicular_distance(points[i], points[start], points[end])
            if dist > max_dist:
                max_dist = dist
                index = i
        if max_dist > epsilon:
            rdp_recurse(points, start, index, epsilon, result)
            result.append(points[index])
            rdp_recurse(points, index, end, epsilon, result)

    result = [points[0]]
    rdp_recurse(points, 0, len(points) - 1, epsilon, result)
    result.append(points[-1])
    return result

def main():
    image = cv2.imread('C:\\Users\\Utsav\\OneDrive\\Desktop\\3426.jpg', cv2.IMREAD_GRAYSCALE)

    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_points = [contour.reshape(-1, 2) for contour in contours]

    epsilon = 7.0  # Adjust this value based on your needs

    simplified_contours = []
    for contour in contour_points:
        simplified_contour = rdp(contour, epsilon)
        simplified_contours.append(np.array(simplified_contour, dtype=np.int32))

    blank_image = np.zeros_like(binary_image)

    for simplified_contour in simplified_contours:
        cv2.drawContours(blank_image, [simplified_contour], -1, (255, 255, 255), 1)

    plt.imshow(blank_image, cmap='gray')
    plt.title('Simplified Contours')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
