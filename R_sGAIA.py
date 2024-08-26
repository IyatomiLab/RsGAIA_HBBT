""""Implementation of Refined Probabilistic Stomach Image Augmentation (R-sGAIA)
"""
from typing import Tuple

import numpy as np
import cv2

from skimage import filters
from skimage import exposure, feature, util
from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import binary_opening, binary_dilation

import random

def normalize_range(X: np.ndarray, a: float=0, b: float=1):
    """Normalize data in range [a, b]"""
    X_min = np.min(X)
    X_max = np.max(X)
    
    normed_X = a + (X - X_min) * (b - a) / (X_max - X_min)
    
    return normed_X

def butterworth_highpass_filter(shape: Tuple, cutoff: float, order: float):
    """Creates a Butterworth high-pass filter.
    
    Args:
        shape (Tuple): shape of the filter (rows, cols)
        cutoff (float): cutoff frequency of the filter
        order (float): order of the filter
    Returns:
        Butterworth high-pass filter
    """
    rows, cols = shape
    x = np.linspace(-0.5, 0.5, cols)
    y = np.linspace(-0.5, 0.5, rows)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    filter_ = 1 / (1 + (cutoff / (distance + 1e-10))**(2 * order))

    return fftshift(filter_)

def compute_high_frequency(I_e: np.ndarray, cutoff: float = 0.02, order: float=3.0):
    """Computes the high-frequency components using a Butterworth high-pass filter
    Args:
        I_e (np.ndarray): contrast-enhanced image I_e(x, y) after histogram equalization
        cutoff (float): cutoff frequency of the filter
        order (float): order of the filter
    Returns:
        high-frequency components
    """
    f_transform = fft2(I_e)

    rows, cols = I_e.shape
    butterworth_filter = butterworth_highpass_filter((rows, cols), cutoff, order)
    high_freq_transform = f_transform * butterworth_filter
    
    high_freq_components = np.abs(ifft2(high_freq_transform))
    high_freq_components = cv2.normalize(high_freq_components, None, 0, 1, cv2.NORM_MINMAX)
    
    return high_freq_components

def calculate_edge_strength(I_e: np.ndarray):
    """Calculate normalized edge strength E(x, y)
    Args:
        I_e (np.ndarray): contrast-enhanced image I_e(x, y) after histogram equalization
    Return:
        E_xy (np.ndarray): normalized edge strength E(x, y)
    """

    # apply canny edge filter (delta_I_e)
    delta_I_e = feature.canny(I_e, sigma=1.6)
    # dilation to enhance the edges
    delta_I_e = binary_dilation(delta_I_e, structure=np.ones((3,3)))

    # apply Butterworth high-pass filter
    # H_e = compute_high_frequency(I_e, cutoff=0.005, order=2.0)
    H_e = filters.butterworth(I_e, high_pass=True, squared_butterworth=True, npad=0)

    # calculate normalized edge strength E(x, y)
    E_xy = (delta_I_e + H_e)/2.0

    return E_xy

def p_sigmoid(E_xy: np.ndarray, gamma=4.0, theta=0.55):
    """Sigmoid function to calculate the probability of a gastric fold region p(x, y)
    Args:
        E_xy (np.ndarray): normalized edge strength E(x, y)
    Return:
        p_xy (np.ndarray): the probability of a gastric fold region p(x, y)
    """
    p_xy =  1 / (1 + np.exp(-gamma*(E_xy - theta)))
    
    return p_xy

def get_gastric_fold_edge_region(p_xy: np.ndarray):
    """Determination of gastric fold edge region G(x, y)
    Args:
        p_xy (np.ndarray): the probability of a gastric fold region p(x, y)
    Returns:
        G_xy (np.ndarray): gastric fold edge region G(x, y)
    """

    # generate a random matrix of the same shape as p_xy
    random_uniform = np.random.rand(*p_xy.shape)
    
    # create G(x, y) based on the comparison with p(x, y)
    G_xy = (p_xy > random_uniform)
    
    # apply morphological opening to clean up the mask
    G_xy = binary_opening(G_xy, structure=np.ones((2, 2)))
    
    # dilation to enhance the mask
    G_xy = binary_dilation(G_xy, structure=np.ones((2, 2)))

    # invert binary mask
    G_xy = util.invert(G_xy)

    return G_xy

def R_sGAIA(img_path: str):
    """Generation of enhanced gastric fold images A(x, y) by R-sGAIA
    Args:
        img_path (str): path to the gastric image
    Return:
        A_xy (np.ndarray): the enhanced gastric fold image by R-sGAIA
    """
    # read image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # apply histogram equalization
    I_e = exposure.equalize_hist(img)

    # calculation of edge strength E(x, y) (Step 1)
    E_xy = calculate_edge_strength(I_e)

    # calculation of the probability of a gastric fold region p(x, y) (Step 2)
    p_xy = p_sigmoid(E_xy)

    # determination of gastric fold edge region G(x, y) (Step 3)
    G_xy = get_gastric_fold_edge_region(p_xy)*I_e

    # generation of enhanced gastric fold images A(x, y) (Step 4)
    alpha = random.uniform(0.9, 1.0)
    beta = random.uniform(-15, -5)

    A_xy = I_e + alpha*G_xy + beta

    # normalize data to match I_e's range
    A_xy = normalize_range(A_xy, a=I_e.min(), b=I_e.max())

    return A_xy