import numpy as np
import tensorflow as tf
import cv2 as cv
import datetime, time
import threading
import azure.iot.device import IoTHubDeviceClient, Message

CONNECTION_STRING = "HostName=ih-e6c97ac3-fa7f-48af-9174-ca3040dfa00f-1.azure-devices.net;DeviceId=0f143f16-ad75-4ba1-bb2e-8e5399be2145;SharedAccessKey=MN/E+FfquilfhU7llfX/Nh2ibc91nXfhTuKEh1tqORo="
MSG_TXT = '{{"SensorValue": {sensorValue}, "Timestamp": {timestamp}}}'

interval = 5 #seconds

def iothub_client_init():
    client = IoTHubClient.create_from_connection_string(CONNECTION_STRING)
    return client

def sendMessage(client, peopleDetected):
    message = Message(MSG_TXT.format(sensorValue=peopleDetected, timestamp=datetime.datetime.utcnow().isoformat()))
    prop_map = message.properties()
    prop_map.add("DigitalTwins-Telemetry", "1.0")
    prop_map.add("DigitalTwins-SensorHardwareId", "RPi1_OCCUPANCY")
    prop_map.add("CreationTimeUtc", datetime.datetime.utcnow().isoformat())
    client.send_message(message)

def sendMessage(peopleDetected):
    print("Detected " + str(peopleDetected) + (" person" if peopleDetected == 1 else " people"))

# Read the graph.
print("Reading the graph")
with tf.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
print("Loaded the graph")

cap = cv.VideoCapture(0)

print("Creating Session")
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
            print("Getting image")
            ret, img = cap.read()
            print("Reshaping image")
            rows = img.shape[0]
            cols = img.shape[1]
            inp = cv.resize(img, (300, 300))
            inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

            # Run the model
            print("Running inference")
            out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                            sess.graph.get_tensor_by_name('detection_scores:0'),
                            sess.graph.get_tensor_by_name('detection_boxes:0'),
                            sess.graph.get_tensor_by_name('detection_classes:0')],
                        feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

            print("Inference done")
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
