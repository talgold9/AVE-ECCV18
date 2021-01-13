import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import h5py
import sys
import cv2
import pylab
import imageio
# from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import argparse
from math import ceil, floor

parser = argparse.ArgumentParser()
parser.add_argument('--video_file', default='/Users/talgoldfryd/Documents/Startup/shooter-trainer/data/videos_ex/vid2.mp4', help='')
parser.add_argument('--output_path', default='/Users/talgoldfryd/Documents/Startup/shooter-trainer/data/videos_ex/vid2.h5', help='')
args = parser.parse_args()

def video_frame_sample(frame_interval, video_length, sample_num):
    num = []
    for l in range(video_length):

        for i in range(sample_num):
            num.append(int(l * frame_interval + (i * 1.0 / sample_num) * frame_interval))

    return num


base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output) # vgg pool5 features

# path of your dataset
video = cv2.VideoCapture(args.video_file)
num_of_frames = video.get(cv2.CAP_PROP_FRAME_COUNT) # length of video
sample_num = int(video.get(cv2.CAP_PROP_FPS)) # frame number for each second
duration = num_of_frames / sample_num
t = floor(duration)
video.release()



video_features = np.zeros([1, t, 7, 7, 512]) # 10s long video


c = 0

'''feature learning by VGG-net'''
video_index = args.video_file
vid = imageio.get_reader(video_index, 'ffmpeg')
vid_len = num_of_frames
frame_interval = int(vid_len / t)

frame_num = video_frame_sample(frame_interval, t, sample_num)
imgs = []
for i, im in enumerate(vid):
    x_im = cv2.resize(im, (224, 224))
    imgs.append(x_im)
vid.close()
extract_frame = []
for n in frame_num:
    extract_frame.append(imgs[n])

feature = np.zeros(([t, sample_num, 7, 7, 512]))
for j in range(len(extract_frame)):
    y_im = extract_frame[j]

    x = image.img_to_array(y_im)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pool_features = np.float32(model.predict(x))

    tt = int(j / sample_num)
    video_id = j - tt * sample_num
    feature[tt, video_id, :, :, :] = pool_features
feature_vector = np.mean(feature, axis=(1)) # averaging features for 16 frames in each second
video_features[0, :, :, :, :] = feature_vector

# save the visual features into one .h5 file. If you have a very large dataset, you may save each feature into one .npy file
with h5py.File(args.output_path, 'w') as hf:
    hf.create_dataset("dataset", data=video_features)