import json
import os
import cv2
import math
import base

class ConventerFromVideoToImage():


	def __init__(self, base_path):
		self.base_path = base_path
		self.metadata = base.ReadFromJson()

	def ConvertToImage(self):
		for filename in self.metadata.keys():
		    print(filename)
		    if (filename.endswith(".mp4")):
		        tmp_path = os.path.join(self.base_path, base.GetFileName(filename))
		        print('Creating Directory: ' + tmp_path)
		        os.makedirs(tmp_path, exist_ok=True)
		        print('Converting Video to Images...')
		        count = 0
		        video_file = os.path.join(self.base_path, filename)
		        capture = cv2.VideoCapture(video_file)
		        frame_rate = capture.get(5) #frame rate
		        while(capture.isOpened()):
		            frame_id = capture.get(1) #current frame number
		            ret, frame = capture.read()
		            if (ret != True):
		                break
		            if (frame_id % math.floor(frame_rate) == 0):
		                print('Original Dimensions: ', frame.shape)
		                if frame.shape[1] < 300:
		                    scale_ratio = 2
		                elif frame.shape[1] > 1900:
		                    scale_ratio = 0.33
		                elif frame.shape[1] > 1000 and frame.shape[1] <= 1900 :
		                    scale_ratio = 0.5
		                else:
		                    scale_ratio = 1
		                print('Scale Ratio: ', scale_ratio)

		                width = int(frame.shape[1] * scale_ratio)
		                height = int(frame.shape[0] * scale_ratio)
		                dim = (width, height)
		                new_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
		                print('Resized Dimensions: ', new_frame.shape)

		                new_filename = '{}-{:03d}.png'.format(os.path.join(tmp_path, base.GetFileName(filename)), count)
		                count += 1
		                cv2.imwrite(new_filename, new_frame)
		        capture.release()
		        print("Done!")
		    else:
		        continue

