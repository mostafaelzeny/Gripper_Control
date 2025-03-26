#import error.py
import cv2
import numpy as np
import time
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep

factory = PiGPIOFactory()
servo = Servo(18, min_pulse_width=0.0005, max_pulse_width=0.0025, pin_factory=factory)

def nothing(x):
    pass

# Function to calculate the diameter of the detected circle
def find_diameter_circle(circle, pixel_to_cm_ratio):
    if circle is not None:
        x, y, radius = circle
        diameter_in_pixels = 2 * radius
        diameter_in_cm = diameter_in_pixels / pixel_to_cm_ratio
        return diameter_in_pixels, diameter_in_cm
    return 0, 0

# Define HSV color ranges for ripe and immature oranges
ripe_orange_lower = (10, 150, 150)
ripe_orange_upper = (25, 255, 255)
immature_orange_lower = (20, 100, 100)
immature_orange_upper = (50, 255, 255)

# Setting up the video capture
cap = cv2.VideoCapture(0)

# Create a window for trackbars to tune the color detection
cv2.namedWindow("Trackbars")
cv2.createTrackbar("Lower-H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("Lower-S", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("Lower-V", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("Upper-H", "Trackbars", 22, 179, nothing)
cv2.createTrackbar("Upper-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Upper-V", "Trackbars", 255, 255, nothing)

# Define the known diameter of a reference object in cm
KNOWN_DIAMETER_CM = 2.5

# Variable to hold pixel-to-cm ratio
pixel_to_cm_ratio = None
last_update_time = time.time()  # Track the last update time

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame = cv2.resize(frame, (640, 480))
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the trackbar positions for HSV ranges
    lower_h = cv2.getTrackbarPos("Lower-H", "Trackbars")
    lower_s = cv2.getTrackbarPos("Lower-S", "Trackbars")
    lower_v = cv2.getTrackbarPos("Lower-V", "Trackbars")
    upper_h = cv2.getTrackbarPos("Upper-H", "Trackbars")
    upper_s = cv2.getTrackbarPos("Upper-S", "Trackbars")
    upper_v = cv2.getTrackbarPos("Upper-V", "Trackbars")

    lower_orange = np.array([lower_h, lower_s, lower_v])
    upper_orange = np.array([upper_h, upper_s, upper_v])

    ripe_mask = cv2.inRange(hsv_frame, np.array(ripe_orange_lower), np.array(ripe_orange_upper))
    immature_mask = cv2.inRange(hsv_frame, np.array(immature_orange_lower), np.array(immature_orange_upper))
    mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    diameter_cm = 0  # Default diameter

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

        if radius > 10:
            if pixel_to_cm_ratio is None:
                pixel_to_cm_ratio = (2 * radius) / KNOWN_DIAMETER_CM
                print(f"Calibrated: {pixel_to_cm_ratio:.2f} pixels per cm")

            _, diameter_cm = find_diameter_circle((x, y, radius), pixel_to_cm_ratio)

            # Tripling the diameter values
            diameter_cm *= 3

            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
            cv2.putText(frame, f"Diameter: {diameter_cm:.2f} cm", (int(x) - 50, int(y) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Check if 10 seconds have passed since the last update
    current_time = time.time()
    if current_time - last_update_time >= 10:  # 10 seconds
        if 5 < diameter_cm < 13 : 
            t = (-0.136 * diameter_cm) + 2.161
            print("Orange detected! Rotating clockwise...")
            servo.value = 0.6  # Full speed clockwise
            sleep(t)
        
            print("Stopping...")
            servo.value = 0  # Stop
            sleep(2)
        
            print("Rotating counterclockwise...")
            servo.value = -0.6  # Full speed counterclockwise
            sleep(t)
        
            print("Stopping...")
            servo.value = 0  # Stop
            sleep(2)
        else:
            print("No orange detected. Stopping motor.")
            servo.value = 0  # Ensure motor is stopped
            sleep(1)

        # Update the last update time
        last_update_time = current_time

    cv2.imshow("Mask", mask)
    cv2.imshow("Detected Orange", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
