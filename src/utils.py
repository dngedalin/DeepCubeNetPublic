import numpy as np
from hdf5storage import loadmat,savemat
from scipy.interpolate import griddata
import os
from tqdm import tqdm
import keras
import tensorflow as tf
import keras.backend as K
import skimage

def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=K.max(y_true))
def cos_distance(y_true, y_pred):
    def l2_normalize(x, axis):
        norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
        return K.maximum(x, K.epsilon()) / K.maximum(norm, K.epsilon())

    y_true = l2_normalize(y_true, axis=-1)
    y_pred = l2_normalize(y_pred, axis=-1)
    return K.mean(K.sum(y_true * y_pred, axis=-1))
def relRMSE(y_true,y_pred):
    true_norm = K.sqrt(K.sum(K.square(y_true), axis=-1))
    return K.mean(K.sqrt(keras.losses.mean_squared_error(y_true, y_pred))/true_norm)
def SSIM(y_true,y_pred):
    return tf.image.ssim(y_pred,y_true,K.max(y_true))
def spectral_TV(y_true,y_pred):
    return K.mean(K.mean(K.sqrt(K.square(y_pred[:, :, :, 1:] - y_pred[:, :, :, :314]))))