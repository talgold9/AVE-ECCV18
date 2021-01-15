import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set gpu number
import numpy as np
import tensorflow as tf
import vggish_input
import vggish_params
import vggish_slim
import h5py
import argparse
import torchvision
from math import floor

parser = argparse.ArgumentParser()
parser.add_argument('--video_file', default='/Users/talgoldfryd/Documents/Startup/shooter-trainer/data/videos_ex/vid2.mp4', help='')
parser.add_argument('--output_path', default='/Users/talgoldfryd/Documents/Startup/shooter-trainer/data/videos_ex/', help='')
args = parser.parse_args()

# Paths to downloaded VGGish files.
checkpoint_path = 'vggish_model.ckpt'
pca_params_path = 'vggish_pca_params.npz'
vid = torchvision.io.read_video(args.video_file)
num_secs = floor(vid[1].shape[1]/ vid[2]['audio_fps']) # length of the audio sequence. Videos in our dataset are all 10s long.

sr = vid[2]['audio_fps']

# path of audio files and AVE annotation
# audio_dir = "..." # .wav audio files
# lis = os.listdir(audio_dir)

audio_features = np.zeros([1, num_secs, 128])


'''feature learning by VGG-net trained by audioset'''
audio_index = args.video_file# path of your audio files

input_batch = vggish_input.wavfile_to_examples(vid[1].numpy(), sr, num_secs)
np.testing.assert_equal(
    input_batch.shape,
    [num_secs, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS])

# Define VGGish, load the checkpoint, and run the batch through the model to
# produce embeddings.
print('Create audio features')
with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session() as sess:
    vggish_slim.define_vggish_slim()
    vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)
    [embedding_batch] = sess.run([embedding_tensor],
                                 feed_dict={features_tensor: input_batch})
    #print('VGGish embedding: ', embedding_batch[0])
    #print(embedding_batch.shape)
    audio_features[0, :, :] = embedding_batch



# save the audio features into one .h5 file. If you have a very large dataset, you may save each feature into one .npy file
output_file = args.video_file.split('/')[-1].split('.')[0] + '_aud.h5'
with h5py.File(os.path.join(args.output_path, output_file), 'w') as hf:
    hf.create_dataset("dataset",  data=audio_features)
