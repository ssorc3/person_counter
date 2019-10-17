import asyncio
import numpy as np
import tensorflow as tf
import cv2 as cv
import datetime, time
import threading
from azure.iot.device.aio import IoTHubDeviceClient
from azure.iot.device import Message
import os

CONNECTION_STRING = os.environ.get("AZURE_CONNECTION_STRING")
MSG_TXT = '{{"SensorValue": {sensorValue}, "Timestamp": {timestamp}}}'

interval = 5 #seconds

def iothub_client_init():
    client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
    return client

async def sendMessage(client, peopleDetected):
    message = Message(MSG_TXT.format(sensorValue=2, timestamp=datetime.datetime.utcnow().isoformat()))
    message.custom_properties["DigitalTwins-Telemetry"] = "1.0"
    message.custom_properties["DigitalTwins-SensorHardwareId"] = "RPi1_OCCUPANCY"
    await device_client.send_message(message)

async def main():
    # Read the graph.
    print("Reading the graph")
    with tf.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    print("Loaded the graph")

    cap = cv.VideoCapture(0)
    client = iothub_client_init()

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
                sendMessage(client, peopleDetected)
                lastSent = time.time()
    cap.release()
    cv.destroyAllWindows()
    cv.waitKey()    

if __name__ == "__main__":
    asyncio.run(main())