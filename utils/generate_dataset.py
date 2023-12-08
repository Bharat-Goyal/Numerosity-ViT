import cv2
import numpy as np
import random
import os 

def generate_exp2_images(directory='/home/hice1/bgoyal7/scratch/HML/experiment_data/exp2_equal_circumference_circles'):
    for i in range(2, 10):
        ct = 1
        for circumference in [100, 150, 200, 250, 300]:
            for _ in range(1, 5):
                image = np.ones((720, 720), dtype=np.uint8)*255
                circles = []
                cumulative_circumference = 0

                for _ in range(i):
                    radius = int(circumference/(i*2*np.pi))
                    x = random.randint(radius, 720 - radius)
                    y = random.randint(radius, 720 - radius)
                    circle = (x, y, radius)
                    circles.append(circle)
                    cumulative_circumference += 2 * np.pi * radius
                    cv2.circle(image, (x, y), radius, 0, -1)

                filename = f"{i}_{ct}.png"
                loc = f"{directory}/{i}"
                if not os.path.exists(loc):
                    os.makedirs(loc)
                filepath = f"{loc}/{filename}"  # Add this line to include the directory in the filename
                # print(filepath)
                cv2.imwrite(filepath, image)
                ct += 1


          
def generate_exp3_images(directory='/home/hice1/bgoyal7/scratch/HML/experiment_data/exp3_equal_area_diff_shapes'):
    shapes = ['circle', 'square', 'triangle']
    areas = [0.02, 0.04, 0.06, 0.08, 0.1]  # Percentage of image area

    for i in range(1, 10):
        ct = 1
        for area_percentage in areas:
            for _ in range(4):
                image = np.ones((720, 720), dtype=np.uint8) * 255
                cumulative_area = 0
                shape = random.choice(shapes)
                area_per_shape = (area_percentage * 720 * 720) / i
                for _ in range(i):
                    if shape == 'circle':
                        radius = int(np.sqrt(area_per_shape / np.pi))
                        x = random.randint(radius, 720 - radius)
                        y = random.randint(radius, 720 - radius)
                        cv2.circle(image, (x + radius, y + radius), radius, 0, -1)
                        cumulative_area += np.pi * radius * radius
                    elif shape == 'square':
                        side_length = int(np.sqrt(area_per_shape))
                        x = random.randint(0, 720 - side_length)
                        y = random.randint(0, 720 - side_length)
                        cv2.rectangle(image, (x, y), (x + side_length, y + side_length), 0, -1)
                        cumulative_area += side_length * side_length
                    elif shape == 'triangle':
                        side_length = int(np.sqrt(2*area_per_shape))
                        x = random.randint(0, 720 - side_length)
                        y = random.randint(0, 720 - side_length)
                        points = np.array([[x, y], [x + side_length, y], [x + side_length // 2, y + side_length]])
                        cv2.fillPoly(image, [points], 0)
                        cumulative_area += 0.5 * side_length * side_length

                filename = f"{i}_{ct}.png"
                loc = f"{directory}/{i}"
                if not os.path.exists(loc):
                    os.makedirs(loc)
                filepath = f"{loc}/{filename}"
                cv2.imwrite(filepath, image)
                ct += 1
    
if __name__ == '__main__': 
    # generate_exp2_images()
    generate_exp3_images()