from flask import Flask, render_template, Response
import cv2
import threading
import argparse
import datetime
import time
import sys
import dlib
import re
import numpy as np

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

from trackable_object import TrackableObject
from centroid_tracker import CentroidTracker



parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-m', '--model', type=str,
                    default='larva.tflite', help='File path of .tflite file.')
parser.add_argument('-l', '--labelmap', type=str,
                    default='labels.txt', help='File path of labels file.')
parser.add_argument('-v', '--video_path', type=str, default='',
                    help='Path to video. If None camera will be used')
parser.add_argument('-t', '--threshold', type=float,
                    default=0.3, help='Detection threshold')
parser.add_argument('-roi', '--roi_position', type=float,
                    default=0.5, help='ROI Position (0-1)')
parser.add_argument('-la', '--labels', nargs='+', type=str,
                    help='Label names to detect (default="all-labels")')
parser.add_argument('-a', '--axis', default=False, action="store_false",
                    help='Axis for cumulative counting (default=x axis)')
parser.add_argument('-e', '--use_edgetpu',
                    action='store_true', default=True, help='Use EdgeTPU')
parser.add_argument('-s', '--skip_frames', type=int, default=3,
                    help='Number of frames to skip between using object detection model')
parser.add_argument('-sh', '--show', default=True,
                    action="store_false", help='Show output')
parser.add_argument('-sp', '--save_path', type=str, default='',
                    help='Path to save the output. If None output won\'t be saved')
parser.add_argument('--type', choices=['tensorflow', 'yolo', 'yolov3-tiny'],
                    default='tensorflow', help='Whether the original model was a Tensorflow or YOLO model')
args = parser.parse_args()




#Initialize the Flask app
app = Flask(__name__)

model = 'larva_edgetpu.tflite'
enable_edgetpu = True
num_threads = 4


camera = cv2.VideoCapture(1)
#camera = cv2.VideoCapture('test_video.mp4')
#codec = cv2.VideoWriter_fourcc( 'Y', 'U', 'Y', 'V')
#camera.set(6, codec)
camera.set(5, 30)
#camera.set(3, 1280)
#camera.set(4, 720)

if args.save_path:
    width = int(camera.get(3))
    height = int(camera.get(4))
    fps_write = camera.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(args.save_path, cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), fps_write, (width, height))

# Visualization parameters
row_size = 30  # pixels
left_margin = 24  # pixels
text_colour = (0x00, 0x45, 0xFF)
line_colour = (0x00, 0x45, 0xFF)
id_colour = (0x00, 0xFF, 0x7F)
font_size = 1.5
font_thickness = 1
fps_avg_frame_count = 10

# Initialize the object detection model
base_options = core.BaseOptions(
  file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
detection_options = processor.DetectionOptions(
  max_results=20, score_threshold=0.3)
options = vision.ObjectDetectorOptions(
  base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)




@app.route('/')
def index():
    return render_template('index.html')




def count():
    fps_counter, fps = 0, 0
    start_time = time.time()
    
    counter = 0
    total_frames = 0

    ct = CentroidTracker(maxDisappeared=5, maxDistance=200)
    trackers = []
    trackableObjects = {}
    
    
    while True:
        success, image_np = camera.read() # read the camera frame
        if not success:
            break
            
        fps_counter += 1
        image_np = cv2.flip(image_np, 1)
        height, width, _ = image_np.shape
        print('height: %s' % height)
        print('width: %s' % width)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)


        status = "Waiting"
        rects = []
        
        if total_frames % args.skip_frames == 0:
            status = "Detecting"
            trackers = []

            # Perform inference
            #results = detect_objects(interpreter, image_pred, args.threshold, args.type)
                
            # Create a TensorImage object from the RGB image.
            input_tensor = vision.TensorImage.create_from_array(rgb_image)

            # Run object detection estimation using the model.
            detection_result = detector.detect(input_tensor)


            print(len(detection_result.detections))


            for detection in detection_result.detections:
                
                
                bbox = detection.bounding_box
                start_point = bbox.origin_x, bbox.origin_y
                end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
                
                cv2.rectangle(image_np, start_point, end_point, text_colour, 1)

                
                category = detection.classes[0]
                class_name = category.class_name
                score = round(category.score, 2)

                
                x_min = bbox.origin_x
                x_max = bbox.origin_x + bbox.width
                y_min = bbox.origin_y
                y_max = bbox.origin_y + bbox.height
                
                print('xy: ')
                print(x_min)
                print(x_max)
                print(y_min)
                print(y_max)
                print('score: %s' % score)
                # if score > args.threshold:
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(int(x_min), int(y_min), int(x_max), int(y_max))
#                    rect = cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

                tracker.start_track(rgb_image, rect)
                trackers.append(tracker)
                tracker.update(rgb_image)

#                if x_min < width and x_max < width and y_min < height and y_max < height and x_min > 0 and x_max > 0 and y_min > 0 and y_max > 0:
                    # add the bounding box coordinates to the rectangles list
                rects.append((x_min, y_min, x_max, y_max))
                
                
                    
        else:
            status = "Tracking"
            for tracker in trackers:
                # update the tracker and grab the updated position
                tracker.update(rgb_image)
                pos = tracker.get_position()

                # unpack the position object
                x_min, y_min, x_max, y_max = int(pos.left()), int(
                    pos.top()), int(pos.right()), int(pos.bottom())
                
                cv2.rectangle(image_np, (x_min,y_min), (x_max,y_max), text_colour, 1)

                    # add the bounding box coordinates to the rectangles list
                rects.append((x_min, y_min, x_max, y_max))

        objects = ct.update(rects)
        print('rects: %s' % len(rects))
        print('obj: %s' % len(objects))


        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)
            #else:
            if args.axis and not to.counted:
                x = [c[0] for c in to.centroids]
                direction = centroid[0] - np.mean(x)

                if centroid[0] > args.roi_position*width and direction > 0 and np.mean(x) < args.roi_position*width:
                    counter += 1
                    to.counted = True
                elif centroid[0] < args.roi_position*width and direction < 0 and np.mean(x) > args.roi_position*width:
                    counter += 1
                    to.counted = True

            elif not args.axis and not to.counted:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)

                if centroid[1] > args.roi_position*height and direction > 0 and np.mean(y) < args.roi_position*height:
                    counter += 1
                    to.counted = True
                elif centroid[1] < args.roi_position*height and direction < 0 and np.mean(y) > args.roi_position*height:
                    counter += 1
                    to.counted = True

                to.centroids.append(centroid)

            trackableObjects[objectID] = to

            text = "ID {}".format(objectID)
            cv2.putText(image_np, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, id_colour, 2)
            cv2.circle(
                image_np, (centroid[0], centroid[1]), 4, id_colour, -1)

        # Draw ROI line
        if args.axis:
            cv2.line(image_np, (int(args.roi_position*width), 0),
                     (int(args.roi_position*width), height), line_colour, 2)
        else:
            cv2.line(image_np, (0, int(args.roi_position*height)),
                     (width, int(args.roi_position*height)), line_colour, 2)


        # Calculate the FPS
        if fps_counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count/(end_time - start_time)
            start_time = time.time()

                # Show the FPS
        fps_text = 'FPS: {:.1f}'.format(fps)
        text_location = (left_margin, row_size)
        cv2.putText(image_np, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                            font_size, text_colour, font_thickness)


        # display count and status
        count_text = 'Count: {}'.format(counter)
        
        if args.axis:
            cv2.putText(image_np, count_text, text_location + (0,10), cv2.FONT_HERSHEY_PLAIN,
                            font_size, text_colour, font_thickness)
        else:
            cv2.putText(image_np, count_text, (left_margin,row_size*2), cv2.FONT_HERSHEY_PLAIN,font_size, text_colour, font_thickness)
        cv2.putText(image_np, 'Status: ' + status, (left_margin,row_size*3), cv2.FONT_HERSHEY_PLAIN,font_size, text_colour, font_thickness)

       
        if args.save_path:
            out.write(image_np)
        
        total_frames += 1


                # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

        ret, buffer = cv2.imencode('.jpg', image_np)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(count(), mimetype='multipart/x-mixed-replace; boundary=frame')

# check to see if this is the main thread of execution
if __name__ == '__main__':
    # start the flask app
    app.run(host='0.0.0.0', port='8000', debug=True,
        threaded=True, use_reloader=False)
# release the video stream pointer
camera.release()
