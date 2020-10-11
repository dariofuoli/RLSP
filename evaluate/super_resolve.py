import os
import numpy as np
import tensorflow as tf  # tested with tensorflow 1.14
import skvideo.io as io
from skimage.io import imsave
from skimage.transform import rescale
import glob
import time
from scipy.ndimage.filters import gaussian_filter as gf

# disable for evaluation on gpu
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

def myycbcr2rgb(vid):
    r = np.sum(vid * np.array([298.082, 0, 408.583]) / 256, axis=-1, keepdims=True) - 222.921
    g = np.sum(vid * np.array([298.082, -100.291, -208.120]) / 256, axis=-1, keepdims=True) + 135.576
    b = np.sum(vid * np.array([298.082, 516.412, 0]) / 256, axis=-1, keepdims=True) - 276.836

    return np.rint(np.clip(np.concatenate([r, g, b], axis=-1), 0, 255)).astype(np.uint8)

def upsample_cb_cr(vid):
    # rgb to (y)cbcr
    cb = np.sum(vid * np.array([-37.945, -74.494, 112.439]) / 256, axis=-1, keepdims=True) + 128
    cr = np.sum(vid * np.array([112.439, -94.154, -18.285]) / 256, axis=-1, keepdims=True) + 128

    cb_hr = rescale(cb, scale=4.0, order=3, preserve_range=True, multichannel=True)
    cr_hr = rescale(cr, scale=4.0, order=3, preserve_range=True, multichannel=True)

    return cb_hr, cr_hr

    
downscale = True
buffer_len = 20
video_path = "FullHD_sample.mp4"
model_name = "ckpt_rlsp_128"

os.makedirs("results", exist_ok=True)

# load model
tf.reset_default_graph()
path = os.path.join('checkpoints', model_name + '.meta')
meta_graph = tf.train.import_meta_graph(path)

session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
meta_graph.restore(sess=session, save_path=os.path.join('checkpoints', model_name))

# load placeholders
c_input = tf.get_collection('c_input')[0]
c_state_input = tf.get_collection('c_state_input')[0]
c_fb_input = tf.get_collection('c_fb_input')[0]
c_output = tf.get_collection('c_output')[0]
c_state_output = tf.get_collection('c_state_output')[0]

# load video frames
reader = io.vreader(video_path)

# fill buffer
buffer = []
buffer_hr = []
for i in range(buffer_len):

    add_image = next(reader)
    buffer_hr.append(add_image)
    
    # downscale by factor 4 with gaussian smoothing
    if downscale:
        s = 1.5
        add_image = gf(add_image, sigma=[s, s, 0])[0::4, 0::4, :]
        add_image = np.rint(np.clip(add_image, 0, 255)).astype(np.uint8)

    buffer.append(add_image)

print("filled buffer...")


def get_batch():

    out = buffer[0:1+2]
    out_hr = buffer_hr[0:3]
    buffer.pop(0)
    buffer_hr.pop(0)
    add_image = next(reader)
    buffer_hr.append(add_image)
    
    # downscale by factor 4 with gaussian smoothing
    if downscale:
        s = 1.5
        add_image = gf(add_image, sigma=[s, s, 0])[0::4, 0::4, :]
        add_image = np.rint(np.clip(add_image, 0, 255)).astype(np.uint8)

    buffer.append(add_image)

    return np.array(out), np.array(out_hr)


# get super-resolved frames
i = 0
while True:

    print(i)

    # get batch
    batch, batch_hr = get_batch()
    if i == 0:
        shape = batch[0].shape
        c = np.zeros((1, shape[0], shape[1], 128))
        y = np.zeros((1, 4 * shape[0], 4 * shape[1], 1))

    y, c = session.run([c_output, c_state_output], feed_dict={c_input: batch[None, :],
                                                              c_state_input: c,
                                                              c_fb_input: y,
                                                              })

    # add cb and cr channels to y
    print(y.shape)
    cb, cr = upsample_cb_cr(batch[1, :])
    print(cb.shape, cr.shape)
    ycbcr = np.rint(np.clip(np.concatenate([y[0], cb, cr], axis=-1), 0, 255))
    rgb = myycbcr2rgb(ycbcr)

    # save frame 
    imsave('results/' + str(i).zfill(8) + '.png', rgb)
    imsave('results/' + str(i).zfill(8) + '_lr.png', batch[1,:])

    # increment index
    i+=1
