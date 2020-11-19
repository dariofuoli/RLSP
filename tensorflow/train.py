# train logic and dataloading to be implemented
# use model.py to generate the RNN object, which implements all the needed components and returns a session

import tensorflow as tf
import model
import numpy as np


# build model
mo = model.Rnn(model_name="model_name", summary_name="summary_name")
session = mo.build(layers=7, 
                   filters=128,
                   state_depth=128, 
                   seqlen=10, 
                   m_frames=1, 
                   p_frames=1, 
                   in_ch=3)

# initial values
IMAGESIZE = 256
BATCHSIZE = 4
init_state = np.zeros((BATCHSIZE, int(IMAGESIZE/4), int(IMAGESIZE/4), 128))
init_fb = np.zeros((BATCHSIZE, IMAGESIZE, IMAGESIZE, 1))

# train loop
for step in range(10**6):

    # train
    x_train_batch = None
    y_train_batch = None

    _, loss, psnr, summary = session.run([mo.optimizer, mo.loss_y, mo.psnr_y, mo.merged],
                                                feed_dict={
                                                    mo.learning_rate: 10**-4,
                                                    mo.s_init_state: init_state,
                                                    mo.s_init_fb: init_fb,
                                                    mo.s_input: x_train_batch,
                                                    mo.s_target: y_train_batch,
                                                })
    
    mo.train_writer.add_summary(summary, step)
    
    # validation
    x_val_batch = None
    y_val_batch = None

    loss, psnr, summary = session.run([mo.loss_y, mo.psnr_y, mo.merged],
                                                feed_dict={
                                                    mo.s_init_state: init_state,
                                                    mo.s_init_fb: init_fb,
                                                    mo.s_input: x_val_batch,
                                                    mo.s_target: y_val_batch,
                                                })
    
    mo.val_writer.add_summary(summary, step)

    # save
    mo.saver.save(sess=session, save_path="logs/checkpoints", global_step=step)
