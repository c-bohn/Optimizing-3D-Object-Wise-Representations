import numpy as np
from PIL import Image, ImageEnhance
from skimage import measure
from skimage.transform import resize
# from mayavi import mlab
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from os import remove


# --------------------------------------------------
# --- Visualize 2D images

def show_image(img, path=None):
    """
    :param img:
    :param path:
    :return:
    """

    img = Image.fromarray((255*img).astype(np.uint8))

    if path is None:
        img.show()
    else:
        try:
            img.save(path)
            print('Saved image to {}.'.format(path))
        except IOError:
            print('Path {} does not exist.'.format(path))


def show_img_list(img_list, path=None, axis=-1):
    """

    :param img_list:      [(H, W, C) for all scenes]
    :param path:
    :return:
    """

    if axis == -1:
        imgs = img_list
        n_rows, n_columns, h, w, c = imgs.shape

        imgs = np.transpose(imgs, (0, 2, 1, 3, 4))
        imgs = np.reshape(imgs, (n_rows * h, n_columns * w, c))

    elif axis == 0:       # multiple scenes
        imgs = np.stack(img_list, axis=axis)
        n, h, w, c = imgs.shape
        imgs = np.reshape(imgs, (n * h, w, c))

    show_image(imgs, path)


def combine_images(img_path_list, out_path, remove_imgs=1, axis=0, layout=None):
    """"
    """

    if len(img_path_list) == 0:
        print('[WARNING] viz.combine_images(): No images in list.')
        return

    if layout is not None:
        np.product(layout) == len(img_path_list)

    img_list = []

    for path in img_path_list:
        img = Image.open(path)
        img_list.append(np.array(img))
        if remove_imgs == 1:
            remove(path)

    if layout is None:
        imgs = np.stack(img_list, axis=axis)
        if axis == 0:
            n, h, w, c = imgs.shape
            imgs = np.reshape(imgs, (n*h, w, c))
        elif axis == 1:
            h, n, w, c = imgs.shape
            offset = 5
            imgs = np.concatenate([imgs, np.zeros((h, n, offset, c))], axis=2)
            imgs = np.reshape(imgs, (h, n*(w+offset), c))
    else:
        imgs = np.stack(img_list, axis=0)
        n, h, w, c = imgs.shape
        shape_new = tuple(layout+[h, w, c])
        imgs = np.reshape(imgs, shape_new)
        if axis == 0:
            imgs = np.transpose(imgs, (0, 2, 1, 3, 4))
            imgs = np.reshape(imgs, (layout[0] * h, layout[1] * w, c))

    imgs = np.uint8(imgs)
    img = Image.fromarray(imgs)
    img.save(out_path)


def color_mapping(img, cmap, col_scale=1.):
    """
    :param img:         (..., W, H, 1)
    :param cmap:        <string>
    :param col_scale:   <float>
    :return:            (..., W, H, 3)
    """

    if img.shape[-1] == 1:
        img = img[..., 0]
    cm = plt.get_cmap(cmap)
    colored_image = cm(col_scale*img)[..., :3]    # ignore alpha channel

    return colored_image

# --------------------------------------------------
# --- Visualize renderings


def show_depth_list(depth_list, path=None, use_color_map=False):
    """

    :param depth_list:      [(N_depth, H, W, 1) for all objects]
    :param path:
    :return:
    """
    all_depth = np.stack(depth_list)    # (N_obj, N_depth, H, W, 1)
    if len(all_depth.shape) == 4:
        all_depth = np.expand_dims(all_depth, axis=-1)
    N_obj, N_depth, H, W, _ = all_depth.shape

    # # ----------- Test
    #
    # R_t = np.transpose(cnfg['R'])
    # K_inv = np.linalg.inv(cnfg['K'])
    # t = np.expand_dims(cnfg['t'], -1)
    #
    # ex = all_depth[0, -1, ...]
    # ex_depth = ex.flatten()
    # X, Y = np.meshgrid(np.linspace(-1., 1., H), np.linspace(-1., 1., W))
    # X = np.reshape(X, [-1])
    # Y = np.reshape(Y, [-1])
    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.set_xlabel("x-axis")
    # ax.set_ylabel("y-axis")
    # ax.set_zlabel("z-axis")
    #
    # # points_3d = np.stack([X, Y, ex_depth], axis=1)
    # # ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], color='g')
    #
    # coords_3d = np.stack([X, Y, np.ones_like(ex_depth)], axis=1)
    # points_3d = np.expand_dims(ex_depth, axis=1) * coords_3d
    # ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
    #
    # plt.show()
    # # --------------

    all_depth = np.swapaxes(all_depth, 1, 2)
    all_depth = np.reshape(all_depth, (N_obj*H, N_depth*W, 1))

    # Normalize/ Clip
    # all_depth = np.clip(all_depth, 0., 12.)     # for nicer grey-scale color values -> scene
    # TODO: uncomment
    all_depth = np.clip(all_depth, 4., 7.5)  # for nicer grey-scale color values -> single deepsdf object
    all_depth = (all_depth-np.min(all_depth))/(np.max(all_depth)-np.min(all_depth))

    # Coloring
    if use_color_map:
        all_depth = 255*color_mapping(all_depth, 'gist_rainbow')
        # all_depth = 255*color_mapping(all_depth+0.5, 'bwr')
    else:
        all_depth = 255*np.repeat(all_depth, 3, axis=-1)
    all_depth = all_depth.astype(np.uint8)

    img = Image.fromarray(all_depth)

    if path is None:
        img.show()
    else:
        try:
            img.save(path)
            print('Saved depth image to {}.'.format(path))
        except IOError:
            print('Path {} does not exist.'.format(path))


def show_occ_list(occ_list, path=None, axis=1):
    """

    :param occ_list:      [(H, W[, 1]) for all objects]
    :param path:
    :return:
    """
    # occs = np.stack(occ_list)    # (N_obj, H, W, 1)
    # N_obj, H, W = occs.shape[0:3]
    # print('Viz.show_occ_lis, occs.min/max', np.min(occs), np.max(occs), occs.shape)
    #
    # occs = np.reshape(occs, (N_obj*H, W, 1))
    # occs = np.repeat(occs, 3, axis=-1)
    # occs = (255*occs).astype(np.uint8)

    occs = np.stack(occ_list, axis=axis)    # (H, N_obj, W, 1)

    if axis == 0:
        n, h, w, _ = occs.shape
        occs = np.reshape(occs, (n * h, w, 1))
    elif axis == 1:
        h, n, w, c = occs.shape
        offset = 5
        if n == 1:
            offset=0
        occs = np.concatenate([occs, np.zeros((h, n, offset, c))], axis=2)
        occs = np.reshape(occs, (h, n * (w + offset), c))

    occs = np.repeat(occs, 3, axis=-1)
    occs = (255*occs).astype(np.uint8)

    img = Image.fromarray(occs)

    if path is None:
        img.show()
    else:
        try:
            img.save(path)
            print('Saved occlusion image to {}.'.format(path))
        except IOError:
            print('Path {} does not exist.'.format(path))


def create_normal_img(depth_img, th=0.1):

    if depth_img.shape[-1] != 1:
        depth_img = np.expand_dims(depth_img, axis=-1)
    N_img, H, W, _ = depth_img.shape

    z = np.zeros((N_img, H+2, W+2, 1))
    z[:, 1:H+1, 1:W+1] = depth_img

    dzdx = (z[:, 2:, :] - z[:, :H, :]) / 2.0
    dzdy = (z[:, :, 2:] - z[:, :, :W]) / 2.0
    dzdx = dzdx[:, :, 1:W+1]
    dzdy = dzdy[:, 1:H+1, :]

    dzdx = np.clip(dzdx, -th, th)
    dzdy = np.clip(dzdy, -th, th)
    direction = np.concatenate([-dzdx, -dzdy, th*np.ones_like(dzdx)], axis=-1)

    magnitude = np.sqrt(np.sum(np.square(direction), axis=-1, keepdims=True))
    normal = direction / magnitude

    normal = 0.5 + 0.5*normal
    return normal


# --------------------------------------------------
# --- Visualize slices

def slice_coloring(sdf_values, col_scale=5.):
    """
    :param sdf_values:
    :return:
    """
    if sdf_values.shape[-1] == 1:
        sdf_values = sdf_values[..., 0]
    ones = np.ones_like(sdf_values)
    out = np.clip(-col_scale * sdf_values, 0., 1.)
    inside = np.clip(col_scale * sdf_values, 0., 1.)
    img = np.stack([ones - out, ones - inside - out, ones - inside], axis=-1)

    return img


def show_slice_list(gt_slices=None, pred_slices=None, path=None):
    """

    :param gt_slices:       [(3, W, H, 3)]
    :param pred_slices:     [(3, W, H, 3)]
    :param path:
    :return:
    """
    def format_slice_imgs(slices):
        slices = np.stack(slices, axis=0)
        if len(slices.shape)==4:
            slices = np.expand_dims(slices, axis=0)
        N_obj, N_axes, H, W, _ = slices.shape
        slices = np.transpose(slices, axes=[0, 2, 1, 3, 4])
        slices = np.reshape(slices, (N_obj * H, N_axes * W, 3))
        return slices

    if pred_slices is None:
        print('Viz.show_slice_list: prediction is None')
        return

    pred_slices = 255.*format_slice_imgs(pred_slices)
    if gt_slices is not None:
        offset = 5
        border = np.zeros((pred_slices.shape[0], offset, 3))
        gt_slices = format_slice_imgs(gt_slices)
        gt_slices = 255*resize(gt_slices, pred_slices.shape[:2])

        all_slices = np.concatenate([gt_slices, border, pred_slices], axis=1)
    else:
        all_slices = pred_slices

    all_slices = all_slices.astype(np.uint8)

    img = Image.fromarray(all_slices)
    if path is None:
        img.show()
    else:
        try:
            img.save(path)
            print('Saved slice image to {}.'.format(path))
        except IOError:
            print('Path {} does not exist.'.format(path))


# --------------------------------------------------
# --- Visualize 3D shape

def save_off(path, verts, faces):

    assert(verts.shape[1]==3)
    assert(faces.shape[1]==3)

    verts = np.asarray(verts)
    faces = np.asarray(faces)

    with open(path, 'w') as file:

        file.write('OFF\n')
        file.write(str(verts.shape[0]) + " " + str(faces.shape[0]) + ' 0\n')

        for i in range(verts.shape[0]):
            v=verts[i]
            file.write(str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + "\n")

        for i in range(faces.shape[0]):
            f=faces[i]
            file.write("3 " + str(f[0]) + " " + str(f[1]) + " " + str(f[2]) + "\n")


def show_mesh(volume, path=None, path_obj=None):
    """

    :param volume:     [(X, Y, Z)]
    :param path:
    :return:
    """

    # volume = np.transpose(volume, axes=(0, 2, 1))
    #
    # # Matplotlib
    # X, Y, Z = volume.shape
    # verts, faces, normals, values = measure.marching_cubes_lewiner(volume, 0, spacing=(1./X, 1./Y, 1./Z))
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # mesh = Poly3DCollection(verts[faces])
    # mesh.set_edgecolor('k')
    # ax.add_collection3d(mesh)
    #
    # ax.set_xlabel("x-axis")
    # ax.set_ylabel("y-axis")
    # ax.set_zlabel("z-axis")
    #
    # plt.tight_layout()
    # if path is None:
    #     plt.show()
    # else:
    #     plt.savefig(path)

    # Mayavi
    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(volume, 0)
        mlab.clf()
        obj = mlab.triangular_mesh([vert[0] for vert in verts],
                                   [vert[1] for vert in verts],
                                   [vert[2] for vert in verts],
                                   faces,
                                   colormap='viridis')
        mlab.axes(obj, x_axis_visibility=False, y_axis_visibility=False, z_axis_visibility=False,)

        # Save
        if path is None and path_obj is None:
            mlab.show()
        if path is not None:
            try:
                mlab.savefig(path)
                print('Saved mesh image to {}.'.format(path))
            except IOError:
                print('Path {} does not exist.'.format(path))
        if path_obj is not None:
            save_off(path_obj, verts, faces)
    except ValueError as e:
        print('[ERROR] viz.show_mesh(): ', e)
        try:
            mlab.savefig(path)
            print('Saved mesh image to {}.'.format(path))
        except IOError:
            print('Path {} does not exist.'.format(path))


def show_pnt_cloud(pnts):
    """

    :param pnts:    (BS, N, [D], 3)
    :param cnfg:
    :return:
    """
    pnts = pnts.reshape((-1, 3))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")

    ax.scatter(pnts[:, 0], pnts[:, 1], pnts[:, 2])

    plt.show()


# --------------------------------------------------
# --- Visualize latent embedding

def show_latent_hist(z_list, path=None):
    """

    :param z_list: [(N_obj, D_lat)]
    :return:
    """

    z_list = np.stack(z_list, axis=0)
    x = np.tile(np.arange(z_list.shape[1]), z_list.shape[0])
    y = z_list.flatten()
    plt.hist2d(x, y, bins=[z_list.shape[1], 20], range=[[0, 8], [np.min(z_list), np.max(z_list)]])

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def create_sprite_image(images):
    """
    From 'http://www.pinchofintelligence.com/simple-introduction-to-tensorboard-embedding-visualisation/'
    Returns a sprite image consisting of images passed as argument. Images should be count x width x height
    """

    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots, 3))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h, j * img_w:(j + 1) * img_w, :] = this_img

    spriteimage = Image.fromarray(spriteimage.astype(np.uint8))
    enhancer = ImageEnhance.Brightness(spriteimage)
    spriteimage = enhancer.enhance(1.8)

    return spriteimage
