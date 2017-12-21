# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help='path to Caffe deploy prototxt file'
	'MobileNetSSD_deploy.prototxt for Caffe model or '
	'ssd_mobilenet_v1_coco.pbtxt from opencv_extra for TensorFlow model')
ap.add_argument("-m", "--model", required=True,
	help='path to Caffe pre-trained model'
	'MobileNetSSD_deploy.caffemodel for Caffe model or '
	'frozen_inference_graph.pb from TensorFlow.')
ap.add_argument("-f", "--framework", required=True,
	help='Caffe or TensorFlow'
	'C for Caffe or '
	'T for TensorFlow.')
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
if args["framework"]=='C':
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
else:
	CLASSES = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
	
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
if args["framework"]=='C':
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
else:
	net = cv2.dnn.readNetFromTensorflow(args["model"], args["prototxt"])
print("[INFO] model loaded...")
# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
temp = vs.read()
defautlWidth = 1280
temp = imutils.resize(temp, width=defautlWidth)
height, width = temp.shape[:2]
out = cv2.VideoWriter('output' + str(time.time()) + '.avi',fourcc, 20.0, (width,height))

inWidth = 300
inHeight = 300
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	print("[PROCESS] get frame...")
	frame = vs.read()
	frame = imutils.resize(frame, width=defautlWidth)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	#blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
	blob = cv2.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), (meanVal, meanVal, meanVal))
	print("[PROCESS] created blob...")
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	print("[PROCESS] created input...")
	detections = net.forward()
	print("[PROCESS] img forwarded...")

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	# write the flipped frame
	out.write(frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
out.release()
vs.stop()
cv2.destroyAllWindows()
