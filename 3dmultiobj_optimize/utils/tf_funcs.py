import tensorflow as tf


# --------------------------------------------------
# --- TF Dataset Generator

def wrapped_generate_input_shapedec(dataset):
    def f(idx):
        return tf.py_function(func=dataset.prepare_input,
                              inp=[idx],
                              #     scene_id, rgb, obj_sdf, obj_slice_imgs
                              Tout=(tf.int32, tf.float32, tf.float32, tf.uint8))
    return f


def wrapped_generate_input_multiobj(dataset):
    def f(idx):
        return tf.py_function(func=dataset.prepare_input,
                              inp=[idx],
                              #     scene_id, rgb_in, rgb_out, depth, msk, gt_z_extr
                              Tout=(tf.int32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
    return f


def get_data_iterator(data, cnfg_training, type='multiobj', shuffle=0):
    if type == 'multiobj':
        map_func = wrapped_generate_input_multiobj(data)
    elif type == 'shapedec':
        map_func = wrapped_generate_input_shapedec(data)
    else:
        print('[ERROR] tf_funcs.get_data_iterator(): Unknown type {}'.format(type))

    dataset_tf = tf.data.Dataset.from_generator(data.generator, tf.int32, tf.TensorShape([]))
    if shuffle:
        dataset_tf = dataset_tf.shuffle(data.get_size(), reshuffle_each_iteration=True)
    dataset_tf = dataset_tf.map(map_func)
    dataset_tf = dataset_tf.batch(cnfg_training['batch_size'])
    iterator = iter(dataset_tf)

    return dataset_tf, iterator


# --------------------------------------------------
# --- Update Parameters

def get_mon_weight(epoch, start, end, max_epoch):
    """

    :param step:    current training iteration
    :return:
    """
    w = tf.cast(epoch, tf.float32) * (end - start) / max_epoch + start
    w = tf.clip_by_value(
        t=w,
        clip_value_min=min(start, end),
        clip_value_max=max(start, end),
        name='clip_weight'
    )
    return w


# --------------------------------------------------
# --- General Functions

def get_number_of_trainable_parameters():
    """
    Return number of trainable parameters;
    - sum over multiplied dimensions for all trainable variables
    """
    total_parameters = 0
    for variable in tf.compat.v1.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters

    output = "Total number of trainable parameters: {}".format(total_parameters)
    return output


# --------------------------------------------------
# --- Task specific Functions

def gaussian_blur(img, kernel_size=11, sigma=5):
    def gauss_kernel(channels, ks, sig):
        ax = tf.range(-ks // 2 + 1.0, ks // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sig ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

    gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]

    k = int((kernel_size - 1) / 2)
    paddings = tf.constant([[0, 0, ], [k, k, ], [k, k, ], [0, 0, ]])
    img = tf.pad(img, paddings, "REFLECT")

    return tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1, 1],
                                  padding='VALID', data_format='NHWC')


def sampling(args):
    """
    Sample from unit Gaussian (-> apply reparametrization trick)
    :param args:        (BS, [N,] 2, D)
        - z_mean:       (BS, [N,] D),   mean of sample
        - z_log_var:    (BS, [N,] D),   log variance of sample
    :return:            (BS, D),        sampled latent vector
    """

    z_mean = args[..., 0, :]
    z_log_var = args[..., 1, :]

    # sample from normal distribution with default settings mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=tf.shape(z_mean))

    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def create_normal_img(depth_img, th=0.1):
    """
    Generate normal map based on depth map.
    :param depth_img:   (BS, [N_img,] H, W [,1])
    :param th:          <int>, threshold to clip normal vector (for better visualization)
    :return:
    """

    if depth_img.shape[-1] != 1:
        depth_img = tf.expand_dims(depth_img, axis=-1)  # (BS, [N_img,] H, W, 1)
    # if len(depth_img.shape) > 4:
    #     depth_img = tf.reshape(depth_img, (-1, self.img_size, self.img_size, 1))  # (BS', H, W, 1)
    assert(len(depth_img.shape) >= 4)

    z = depth_img

    # dzdx = (z[:, 2:, :] - z[:, :-2, :]) / 2.0
    # dzdy = (z[:, :, 2:] - z[:, :, :-2]) / 2.0
    # dzdx = dzdx[:, :, 1:-1]
    # dzdy = dzdy[:, 1:-1, :]
    dzdx = (z[..., 2:, :, :] - z[..., :-2, :, :]) / 2.0
    dzdy = (z[..., :, 2:, :] - z[..., :, :-2, :]) / 2.0
    dzdx = dzdx[..., :, 1:-1, :]
    dzdy = dzdy[..., 1:-1, :, :]

    dzdx = tf.clip_by_value(dzdx, -th, th)
    dzdy = tf.clip_by_value(dzdy, -th, th)

    direction = tf.concat([-dzdx, -dzdy, th * tf.ones_like(dzdx)], axis=-1)
    magnitude = tf.sqrt(tf.reduce_sum(tf.square(direction), axis=-1, keepdims=True))
    normal = tf.divide(direction, magnitude)

    return normal, dzdx, dzdy


def gen_slice(axis, size=128, n_pnts=4096, slice_pos=0., for_mesh=False):
    """

    :param axis:  option from ['x', 'y', 'z']
    :return:
    """
    with tf.name_scope('gen_slice_{}'.format(axis)):
        if for_mesh:
            N, M = tf.meshgrid(tf.linspace(-1.0, 1.0, size), tf.linspace(-1.0, 1.0, size))
        else:
            M, N = tf.meshgrid(tf.linspace(-1.0, 1.0, size), tf.linspace(-1.0, 1.0, size))
        N = tf.reshape(N, [-1])
        M = tf.reshape(M, [-1])
        N = tf.expand_dims(N, axis=1)
        M = tf.expand_dims(M, axis=1)

        if axis == 'x':
            slice = tf.concat([slice_pos * tf.ones_like(N), M, N], axis=1)
        elif axis == 'y':
            slice = tf.concat([M, slice_pos * tf.ones_like(N), N], axis=1)
        else:
            slice = tf.concat([M, N, slice_pos * tf.ones_like(N)], axis=1)

        assert ((size * size) % n_pnts == 0)
        slice = tf.reshape(slice, [-1, n_pnts, 3])

    return slice


def slice_coloring(sdf_slice, col_scale=5.):
    """
    :param sdf_slice:
    :return:
    """
    with tf.name_scope('sdf_colormap'):
        ones = tf.ones_like(sdf_slice)
        outside = tf.clip_by_value(-col_scale * sdf_slice, 0., 1.)
        inside = tf.clip_by_value(col_scale * sdf_slice, 0., 1.)
        sdfc = tf.concat([ones - outside, ones - inside - outside, ones - inside], axis=-1)

    return sdfc
