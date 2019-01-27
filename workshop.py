import numpy as np
import cv2

# Function to pass to createTrackbar
def nothing(x):
	pass

# Gaussian Blurring
def smoothing():
	cap = cv2.VideoCapture(0)
	while(True):
	    _, frame = cap.read()
	    blurred = cv2.GaussianBlur(frame,(9,9),0)
	    cv2.imshow('frame', frame)
	    cv2.imshow('blurred', blurred)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
	cap.release()
	cv2.destroyAllWindows()

# Threshold effects
def threshold():
	cap = cv2.VideoCapture(0)
	cv2.namedWindow('Thresholds')
	# 0: Binary 1: Binary inverted 2: Threshold truncated
	# 3: Threshold to zero 4: Threshold to zero inverted
	cv2.createTrackbar('type', 'Thresholds', 0, 4, nothing)
	cv2.createTrackbar('value', 'Thresholds', 0, 255, nothing)
	while(True):
	    _, frame = cap.read()
	    # Convert to grayscale image
	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    thresh = cv2.getTrackbarPos('type', 'Thresholds')
	    value = cv2.getTrackbarPos('value', 'Thresholds')
	    _, mask = cv2.threshold(gray, value, 255, thresh)
	    cv2.imshow('frame', frame)
	    cv2.imshow('mask', mask)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
	cap.release()
	cv2.destroyAllWindows()

# inRange filtering
def filter():
	cap = cv2.VideoCapture(0)
	cv2.namedWindow('HSV')
	cv2.createTrackbar('low_H', 'HSV', 0, 255, nothing)
	cv2.createTrackbar('low_S', 'HSV', 0, 255, nothing)
	cv2.createTrackbar('low_V', 'HSV', 0, 255, nothing)
	cv2.createTrackbar('high_H', 'HSV', 255, 255, nothing)
	cv2.createTrackbar('high_S', 'HSV', 255, 255, nothing)
	cv2.createTrackbar('high_V', 'HSV', 255, 255, nothing)
	while(True):
		_, frame = cap.read()
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		low_H = cv2.getTrackbarPos('low_H', 'HSV')
		low_S = cv2.getTrackbarPos('low_S', 'HSV')
		low_V = cv2.getTrackbarPos('low_V', 'HSV')
		high_H = cv2.getTrackbarPos('high_H', 'HSV')
		high_S = cv2.getTrackbarPos('high_S', 'HSV')
		high_V = cv2.getTrackbarPos('high_V', 'HSV')
		low_limit = np.array([low_H,low_S,low_V])
		high_limit = np.array([high_H,high_S,high_V])
		mask = cv2.inRange(hsv, low_limit, high_limit)
		cv2.imshow('frame', frame)
		cv2.imshow('mask', mask)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

# Bitwise AND
def bitwise():
	cap = cv2.VideoCapture(0)
	cv2.namedWindow('HSV')
	cv2.createTrackbar('low_H', 'HSV', 0, 255, nothing)
	cv2.createTrackbar('low_S', 'HSV', 0, 255, nothing)
	cv2.createTrackbar('low_V', 'HSV', 0, 255, nothing)
	cv2.createTrackbar('high_H', 'HSV', 255, 255, nothing)
	cv2.createTrackbar('high_S', 'HSV', 255, 255, nothing)
	cv2.createTrackbar('high_V', 'HSV', 255, 255, nothing)
	while(True):
		_, frame = cap.read()
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		low_H = cv2.getTrackbarPos('low_H', 'HSV')
		low_S = cv2.getTrackbarPos('low_S', 'HSV')
		low_V = cv2.getTrackbarPos('low_V', 'HSV')
		high_H = cv2.getTrackbarPos('high_H', 'HSV')
		high_S = cv2.getTrackbarPos('high_S', 'HSV')
		high_V = cv2.getTrackbarPos('high_V', 'HSV')
		low_limit = np.array([low_H,low_S,low_V])
		high_limit = np.array([high_H,high_S,high_V])
		mask = cv2.inRange(hsv, low_limit, high_limit)
		blurred = cv2.GaussianBlur(mask,(5,5),2)
		cv2.imshow('frame', frame)
		cv2.imshow('mask', blurred)
		bit = cv2.bitwise_and(frame, frame, mask = blurred)
		cv2.imshow('bitwise', bit)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

# Gaussian Blurring
def canny():
	cap = cv2.VideoCapture(0)
	while(True):
	    _, frame = cap.read()
	    blurred = cv2.Canny(frame,50,100)
	    cv2.imshow('frame', frame)
	    cv2.imshow('blurred', blurred)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
	cap.release()
	cv2.destroyAllWindows()

def main():
	#smoothing()
	#threshold()
	#filter()
	#bitwise()
	canny()

if __name__ == '__main__':
	main()