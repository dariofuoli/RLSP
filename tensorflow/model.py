import tensorflow as tf
import os


# RGB -> Y transform 
def myrgb2y(im):
    return tf.reduce_sum(im*tf.constant([65.738, 129.057, 25.064], dtype=tf.float32)/256, axis=-1, keep_dims=True) + 16


# define model: first define cell, then create model class
def cell(c_in, c_state_in, c_fb_in, layers, filters, state_depth, reuse, res_idx=None):

    l_count = 0

    # concatenation of input/states/feedback/...
    if res_idx is None:
        res = tf.identity(tf.concat(tf.unstack(c_in, axis=1), axis=-1))
    else:
        res = tf.identity(c_in[:, res_idx])
    layer = tf.concat(tf.unstack(c_in, axis=1), axis=-1)
    layer = tf.concat([layer, c_state_in, tf.space_to_depth(c_fb_in, block_size=4)], axis=-1)
    layer = tf.layers.conv2d(inputs=layer, filters=filters, kernel_size=3, padding='same',
                             activation=tf.nn.relu, reuse=reuse, name="layer" + str(l_count))
    l_count += 1
    print(layer)

    for j in range(layers - 3):
        layer = tf.layers.conv2d(inputs=layer, filters=filters, kernel_size=3, padding='same',
                                 activation=tf.nn.relu, reuse=reuse, name="layer" + str(l_count))
        l_count += 1
        print(layer)

    layer = tf.layers.conv2d(inputs=layer, filters=filters, kernel_size=3, padding='same',
                             activation=tf.nn.relu, reuse=reuse, name="layer" + str(l_count))
    l_count += 1
    print(layer)

    # seperate state convolution
    c_state_out = tf.layers.conv2d(inputs=layer, filters=state_depth, kernel_size=3, padding='same',
                                   activation=tf.nn.relu, reuse=reuse, name="layer" + str(l_count))
    l_count += 1
    print(c_state_out)

    # last layer of cell
    layer = tf.layers.conv2d(inputs=layer, filters=16, kernel_size=3, padding='same',
                             activation=None, reuse=reuse, name="layer" + str(l_count))
    l_count += 1
    print(layer)

    # using NN upscaling
    res_add = tf.ones_like(layer) * myrgb2y(res)  # tf.tile(input=myrgb2y(res), multiples=[1, 1, 1, 16])

    print(res_add)

    layer = tf.add(res_add, layer)
    print(layer)

    c_output = tf.depth_to_space(layer, block_size=4)
    print(c_output)

    return c_output, c_state_out, res_add


class Rnn:

    def __init__(self, model_name, summary_name):

        self.model_name = model_name
        self.summary_name = summary_name
        self.c_input = None
        self.c_state_input = None
        self.c_fb_input = None
        self.c_output = None
        self.c_state_output = None
        self.c_res_add = None

        self.learning_rate = None
        self.s_input = None
        self.s_target = None
        self.s_init_state = None
        self.s_init_fb = None
        self.outputs = None
        self.states = None
        self.s_res_add = None
        self.loss = None
        self.psnr = None
        self.loss_y = None
        self.psnr_y = None
        self.optimizer = None
        self.merged = None
        self.saver = None
        self.train_writer = None
        self.val_writer = None

    def build(self, layers, filters, state_depth, seqlen, m_frames, p_frames, in_ch):

        # m_frames == "number of minus (t_-1, t_2, ...) frames to add per step"
        # p_frames == "number of plus (t_+1, t_+2, ...) frames to add per step"

        # reset default graph if there is one
        tf.reset_default_graph()

        # set cell placeholders
        self.c_input = tf.placeholder(tf.float32, [None, m_frames + p_frames + 1, None, None, in_ch], name="c_input")
        self.c_state_input = tf.placeholder(tf.float32, [None, None, None, state_depth], name="c_state_input")
        self.c_fb_input = tf.placeholder(tf.float32, [None, None, None, 1], name="c_fb_input")

        # set sequence placeholders
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.s_input = tf.placeholder(tf.float32, [None, seqlen + m_frames + p_frames, None, None, in_ch], name="s_input")
        self.s_target = tf.placeholder(tf.float32, [None, seqlen, None, None, in_ch], name="s_target")
        self.s_init_state = tf.placeholder(tf.float32, [None, None, None, state_depth], name="init_state")
        self.s_init_fb = tf.placeholder(tf.float32, [None, None, None, 1], name="s_init_fb")

        # define cell network
        self.c_output, self.c_state_output, self.c_res_add = cell(c_in=self.c_input,
                                                                  c_state_in=self.c_state_input,
                                                                  c_fb_in=self.c_fb_input,
                                                                  layers=layers,
                                                                  filters=filters,
                                                                  state_depth=state_depth,
                                                                  res_idx=m_frames,
                                                                  reuse=False)

        # define unrolled network
        out_list = []
        state_list = []
        for i in range(m_frames, seqlen + m_frames):
            if i == m_frames:
                out_list.append(self.s_init_fb)
                state_list.append(self.s_init_state)

            s_out, s_state, _ = cell(c_in=self.s_input[:, i-m_frames:i+p_frames+1],
                                     c_state_in=state_list[-1],
                                     c_fb_in=out_list[-1],
                                     layers=layers,
                                     filters=filters,
                                     state_depth=state_depth,
                                     res_idx=m_frames,
                                     reuse=True)
            out_list.append(s_out)
            state_list.append(s_state)

        self.outputs = tf.transpose(out_list, perm=[1, 0, 2, 3, 4], name='outputs')[:, 1:]
        print(self.outputs)
        self.states = tf.transpose(state_list, perm=[1, 0, 2, 3, 4], name='states')[:, 1:]
        print(self.states)

        # define loss and optimizer
        self.loss_y = tf.losses.mean_squared_error(labels=myrgb2y(self.s_target), predictions=self.outputs)
        self.psnr_y = tf.constant(10, dtype=tf.float32) * tf.log(
            tf.constant(255 ** 2, dtype=tf.float32) / self.loss_y) / tf.log(tf.constant(10, dtype=tf.float32))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_y)

        # define summaries
        tf.summary.scalar(name='loss_y', tensor=self.loss_y)
        tf.summary.scalar(name='psnr_y', tensor=self.psnr_y)
        self.merged = tf.summary.merge_all()

        # create saver
        self.saver = tf.train.Saver(max_to_keep=10**3)

        # start session and initialize variables
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
        config.allow_soft_placement = True
        session = tf.Session(config=config)
        session.run(tf.global_variables_initializer())

        # set summary paths and location for graph
        self.train_writer = tf.summary.FileWriter(
            os.path.join('logs/' + self.model_name + "_summaries" + '/' + self.summary_name + '/train'),
            session.graph)
        self.val_writer = tf.summary.FileWriter(os.path.join('logs/' + self.model_name + "_summaries" + '/' + self.summary_name + '/val'))

        # add variables to collection to retrieve when loading saved network
        tf.add_to_collection('c_input', self.c_input)
        tf.add_to_collection('c_state_input', self.c_state_input)
        tf.add_to_collection('c_fb_input', self.c_fb_input)
        tf.add_to_collection('c_output', self.c_output)
        tf.add_to_collection('c_state_output', self.c_state_output)
        tf.add_to_collection('c_res_add', self.c_res_add)

        tf.add_to_collection('learning_rate', self.learning_rate)
        tf.add_to_collection('s_input', self.s_input)
        tf.add_to_collection('s_target', self.s_target)
        tf.add_to_collection('s_init_state', self.s_init_state)
        tf.add_to_collection('s_init_fb', self.s_init_fb)
        tf.add_to_collection('states', self.states)
        tf.add_to_collection('outputs', self.outputs)
        tf.add_to_collection('s_res_add', self.s_res_add)

        tf.add_to_collection('optimizer', self.optimizer)
        tf.add_to_collection('loss_y', self.loss_y)
        tf.add_to_collection('psnr_y', self.psnr_y)
        tf.add_to_collection('merged', self.merged)

        return session


