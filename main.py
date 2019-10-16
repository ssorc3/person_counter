import numpy as np
import tensorflow as tf
import cv2 as cv
import time

interval = 5 #seconds

def sendMessage(peopleDetected):
    print("Detected " + str(peopleDetected) + (" person" if peopleDetected == 1 else " people"))

# Read the graph.
with tf.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

cap = cv.VideoCapture(0)

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    peopleDetected = 0
    lastSent = 0

    # Read and preprocess an image.
    while(True):
        if (lastSent + 2 < time.time()):
            peopleDetected = 0
            ret, img = cap.read()
            rows = img.shape[0]
            cols = img.shape[1]
            inp = cv.resize(img, (300, 300))
            inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

            # Run the model
            out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                            sess.graph.get_tensor_by_name('detection_scores:0'),
                            sess.graph.get_tensor_by_name('detection_boxes:0'),
                            sess.graph.get_tensor_by_name('detection_classes:0')],
                        feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

            # Visualize detected bounding boxes.
            num_detections = int(out[0][0])
            for i in range(num_detections):
                classId = int(out[3][0][i])
                score = float(out[1][0][i])
                if(score > 0.4 and classId == 1):
                   peopleDetected += 1
            sendMessage(peopleDetected)
            lastSent = time.time()
cap.release()
cv.destroyAllWindows()
cv.waitKey()
