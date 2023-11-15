import cv2
import torch
import numpy as np
import einops
import itertools
from scipy import ndimage
import torch.nn.functional as F
# ===============
# image/video transform
# ===============


def augment_img(img, prob=0.5, tform_op=['all']):
    """
    img data augment with a $op chance

    Args:
        img ([ndarray]): [shape: H*W*C]
        prob (float, optional): [probility]. Defaults to 0.5.
        op (list, optional): ['flip' | 'rotate' | 'reverse']. Defaults to ['all'].
    """
    if 'flip' in tform_op or 'all' in tform_op:
        # flip left-right or flip up-down
        if np.random.rand() < prob:
            img = img[:, ::-1, :]
        if np.random.rand() < prob:
            img = img[::-1, :, :]
    if 'rotate' in tform_op or 'all' in tform_op:
        # rotate 90 / -90 degrees
        if prob/4 < np.random.rand() <= prob/2:
            np.transpose(img, axes=(1, 0, 2))[::-1, ...]  # -90
        elif prob/2 < np.random.rand() <= prob:
            img = np.transpose(
                img[::-1, :, :][:, ::-1, :], axes=(1, 0, 2))[::-1, ...]  # 90

    return img.copy()


def tensor2uint(img):
    # convert torch tensor to uint
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    return np.uint8((img*255.0).round())


def imsave(img, img_path, compress_ratio=0):
    # function: RGB image saving （H*W*3， numpy）
    # compress_ratio: 1-10, higher value, lower quality
    # tip: default $compress_ratio for built-in function cv2.imwrite is 95/100 (higher value, higher quality) and 3/10 (higher value, lower quality) for jpg and png foramt, respectively. Here default value set to no compression

    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
        img = img[:, :, ::-1]
    if img_path.split('.')[-1] in ['jpg', 'jpeg']:
        cv2.imwrite(img_path, img, [
                    cv2.IMWRITE_JPEG_QUALITY, (10-compress_ratio)*10])
    elif img_path.split('.')[-1] in ['png', 'PNG']:
        cv2.imwrite(img_path, img, [
                    cv2.IMWRITE_PNG_COMPRESSION, compress_ratio])
    else:
        cv2.imwrite(img_path, img)


def imsave_n(imgs: list, img_path, axis=1, show_flag=False):
    # save one or more images (np.ndarray, [c,h,w]) to one image
    if imgs[0].ndim == 3:
        # rgb images
        imgs = [np.transpose(img, (1, 2, 0)) for img in imgs]
        result_img = np.concatenate(imgs, axis=axis)
        result_img = result_img[:, :, ::-1]
    elif imgs[0].ndim == 2:
        # gray images
        result_img = np.concatenate(imgs, axis=axis)

    cv2.imwrite(img_path, result_img)

    if show_flag:
        cv2.namedWindow("image", 0)
        cv2.imshow("image", result_img.astype(np.uint8))
        cv2.waitKey(0)


def vidsave_n(imgs: list, img_path, axis=0, show_flag=False):
    # save one or more videos (np.ndarray [n,c,h,w]) to one image
    if imgs[0].ndim == 4:
        # rgb videos
        for i in range(len(imgs)):
            imgs[i] = einops.rearrange(imgs[i], "n c h w->h (n w) c")
        result_img = np.concatenate(imgs, axis=axis)
        result_img = result_img[:, :, ::-1]
    elif imgs[0].ndim == 3:
        # gray videos
        for i in range(len(imgs)):
            imgs[i] = einops.rearrange(imgs[i], "n h w->h (n w)")
        result_img = np.concatenate(imgs, axis=axis)

    cv2.imwrite(img_path, result_img)

    if show_flag:
        cv2.namedWindow("image", 0)
        cv2.imshow("image", result_img.astype(np.uint8))
        cv2.waitKey(0)
def augment_vid(vid, prob=0.5, tform_op=['all']):
    """
    video data transform (data augment) with a $op chance

    Args:
        vid ([ndarray]): [shape: N*H*W*C]
        prob (float, optional): [probility]. Defaults to 0.5.
        op (list, optional): ['flip' | 'rotate' | 'reverse']. Defaults to ['all'].
    """
    if 'flip' in tform_op or 'all' in tform_op:
        # flip left-right or flip up-down
        if np.random.rand() < prob:
            vid = vid[:, :, ::-1, :]
        if np.random.rand() < prob:
            vid = vid[:, ::-1, :, :]
    if 'rotate' in tform_op or 'all' in tform_op:
        # rotate 90 / -90 degrees
        if prob/4 < np.random.rand() <= prob/2:
            np.transpose(vid, axes=(0, 2, 1, 3))[:, ::-1, ...]  # -90
        elif prob/2 < np.random.rand() <= prob:
            vid = np.transpose(
                vid[:, ::-1, :, :][:, :, ::-1, :], axes=(0, 2, 1, 3))[:, ::-1, ...]  # 90
    if 'reverse' in tform_op or 'all' in tform_op:
        # reverse video's frame order
        if np.random.rand() < prob:
            vid = vid[::-1, ...]

    return vid.copy()

# ===============
# padding
# ===============


def circular_padding(tensor, psf_sz):
    '''
    circular padding for image batch
    :param x: img, shape [N,C,H,W]
    :param pad: [H,W]
    :return:
    '''
    x_pad_len, y_pad_len = psf_sz[0]-1, psf_sz[1]-1
    pad_width = (y_pad_len//2, y_pad_len-y_pad_len//2,
                 x_pad_len//2, x_pad_len-x_pad_len//2)
    tensor = F.pad(tensor, pad_width, "circular")
    return tensor


def pad_circular(x, pad):
    """

    :param x: shape [H, W]
    :param pad: int >= 0
    :return:
    """
    x = torch.cat([x, x[0:pad]], dim=0)
    x = torch.cat([x, x[:, 0:pad]], dim=1)
    x = torch.cat([x[-2 * pad:-pad], x], dim=0)
    x = torch.cat([x[:, -2 * pad:-pad], x], dim=1)

    return x


def pad_circular_nd(x: torch.Tensor, pad: int, dim) -> torch.Tensor:
    """
    :param x: shape [H, W]
    :param pad: int >= 0
    :param dim: the dimension over which the tensors are padded
    :return:
    """

    if isinstance(dim, int):
        dim = [dim]

    for d in dim:
        if d >= len(x.shape):
            raise IndexError(f"dim {d} out of range")

        idx = tuple(slice(0, None if s != d else pad, 1)
                    for s in range(len(x.shape)))
        x = torch.cat([x, x[idx]], dim=d)

        idx = tuple(slice(None if s != d else -2 * pad, None if s !=
                    d else -pad, 1) for s in range(len(x.shape)))
        x = torch.cat([x[idx], x], dim=d)
        pass

    return x


# ===============
# image processing
# ===============


def img_blur(img, psf, noise_level=0.01, mode='circular'):
    """
    blur image with psf

    Args:
        img (ndarray): sharp image
        psf (ndarray): coded exposure psf
        noise_level (scalar): noise level
        mode (str): convolution mode, 'circular' | valid'

    Returns:
        x: blurred image
    """
    # convolution
    if mode == 'circular':
        blur_img = ndimage.filters.convolve(
            img, np.expand_dims(psf, axis=2), mode='wrap')
    elif mode == 'valid':
        blur_img = ndimage.filters.convolve(
            img, np.expand_dims(psf, axis=2), mode='constant', cval=0.0)
    else:
        raise NotImplementedError(f'"{mode}" mode is not implemented')

    # add Gaussian noise
    blur_noisy_img = blur_img + \
        np.random.normal(0, noise_level, blur_img.shape)
    return blur_noisy_img.astype(np.float32)


def img_saturation(img, mag_times=1.2, min=0, max=1):
    """
    saturation generation by magnify and clip
    """
    # return np.clip(img*mag_times, min, max)
    return np.clip(img*mag_times, min, max)/mag_times


# ===============
# show and save
# ===============

def tensor2uint(img):
    # convert torch tensor to uint
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    return np.uint8((img*255.0).round())


def imsave(img, img_path, compress_ratio=0):
    # function: RGB image saving （H*W*3， numpy）
    # compress_ratio: 1-10, higher value, lower quality
    # tip: default $compress_ratio for built-in function cv2.imwrite is 95/100 (higher value, higher quality) and 3/10 (higher value, lower quality) for jpg and png foramt, respectively. Here default value set to no compression

    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
        img = img[:, :, ::-1]
    if img_path.split('.')[-1] in ['jpg', 'jpeg']:
        cv2.imwrite(img_path, img, [
                    cv2.IMWRITE_JPEG_QUALITY, (10-compress_ratio)*10])
    elif img_path.split('.')[-1] in ['png', 'PNG']:
        cv2.imwrite(img_path, img, [
                    cv2.IMWRITE_PNG_COMPRESSION, compress_ratio])
    else:
        cv2.imwrite(img_path, img)


def imsave_n(imgs: list, img_path, axis=1, show_flag=False):
    # save one or more images (np.ndarray, [c,h,w]) to one image
    if imgs[0].ndim == 3:
        # rgb images
        imgs = [np.transpose(img, (1, 2, 0)) for img in imgs]
        result_img = np.concatenate(imgs, axis=axis)
        result_img = result_img[:, :, ::-1]
    elif imgs[0].ndim == 2:
        # gray images
        result_img = np.concatenate(imgs, axis=axis)

    cv2.imwrite(img_path, result_img)

    if show_flag:
        cv2.namedWindow("image", 0)
        cv2.imshow("image", result_img.astype(np.uint8))
        cv2.waitKey(0)


def vidsave_n(imgs: list, img_path, axis=0, show_flag=False):
    # save one or more videos (np.ndarray [n,c,h,w]) to one image
    if imgs[0].ndim == 4:
        # rgb videos
        for i in range(len(imgs)):
            imgs[i] = einops.rearrange(imgs[i], "n c h w->h (n w) c")
        result_img = np.concatenate(imgs, axis=axis)
        result_img = result_img[:, :, ::-1]
    elif imgs[0].ndim == 3:
        # gray videos
        for i in range(len(imgs)):
            imgs[i] = einops.rearrange(imgs[i], "n h w->h (n w)")
        result_img = np.concatenate(imgs, axis=axis)

    cv2.imwrite(img_path, result_img)

    if show_flag:
        cv2.namedWindow("image", 0)
        cv2.imshow("image", result_img.astype(np.uint8))
        cv2.waitKey(0)


def img_matrix(imgs, n_row, n_col, margin=0):
    '''
        Arrange a number of images as a matrix.

        positional arguments:
        imgs        N images [H,W,C] in a list
        n_row       number of rows in desired images matrix
        n_col       number of columns in desired images matrix
        margin      Margin between images: integers are interpreted as pixels,floats as proportions. default=0

        reference: https://gist.github.com/pgorczak/95230f53d3f140e4939c
    '''

    n = n_col*n_row
    if len(imgs) != n:
        raise ValueError('Number of images ({}) does not match '
                         'matrix size {}x{}'.format(n_col, n_row, len(imgs)))

    if any(i.shape != imgs[0].shape for i in imgs[1:]):
        raise ValueError('Not all images have the same shape.')

    img_h, img_w = imgs[0].shape[0:2]
    if imgs[0].ndim == 2:
        img_c = 1
    else:
        img_c = imgs[0].shape[2]

    m_x = 0
    m_y = 0

    if isinstance(margin, float):
        m = float(margin)
        m_x = int(m*img_w)
        m_y = int(m*img_h)
    else:
        m_x = int(margin)
        m_y = m_x

    imgmatrix = np.zeros((img_h * n_row + m_y * (n_row - 1),
                          img_w * n_col + m_x * (n_col - 1),
                          img_c),
                         np.uint8)
    imgmatrix = imgmatrix.squeeze()

    imgmatrix.fill(255)

    positions = itertools.product(range(n_col), range(n_row))
    for (x_i, y_i), img in zip(positions, imgs):
        x = x_i * (img_w + m_x)
        y = y_i * (img_h + m_y)
        imgmatrix[y:y+img_h, x:x+img_w, ...] = img

    return imgmatrix
