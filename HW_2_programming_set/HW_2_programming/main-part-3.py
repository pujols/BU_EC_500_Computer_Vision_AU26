import argparse
import random
import os
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl


def Periodic_basis(args, N, u_freq, v_freq, phase):
    """
    Input:
        N: image width & height; you can assume N is an odd number
        u: modulation horizontal frequency
        v: modulation vertical frequency
        phase
    Output:
        I: output image, a 3D matrix of size N-by-N-by-1
    """

    #### Your job 1 starts here ####


    #### Your job 1 ends here ####

    return I


def main(args):

    # Create a periodic image
    if int(args.current_step) == 1:
        N = int(args.N)  # N: image width & height; you can assume N is an odd number
        I = Periodic_basis(args, N, 0, 3, 0)

        fig, ax = plt.subplots()
        ax.imshow(np.transpose((I / 2 + 0.5), (1, 0)), cmap='gray', origin='lower', extent = [-((N - 1) / 2), ((N - 1) / 2), -((N - 1) / 2), ((N - 1) / 2)])
        plt.show()
        plt.close(fig)

    # Create a series of periodic image and average them
    if int(args.current_step) == 2:
        N = int(args.N)
        I = np.zeros((N, N))

        for m in range(int((N - 1) / 4)):
            I = I + Periodic_basis(args, N, 0, m, 0)
        I = I / N

        fig, ax = plt.subplots()
        ax.imshow(np.transpose((I / 2 + 0.5), (1, 0)), cmap='gray', origin='lower',extent = [-((N - 1) / 2), ((N - 1) / 2), -((N - 1) / 2), ((N - 1) / 2)])
        plt.show()
        plt.close(fig)

    # Create a series of periodic image and average them
    if int(args.current_step) == 3:
        N = int(args.N)
        I = np.zeros((N, N))

        for m in range(N):
            I = I + Periodic_basis(args, N, 0, m - (N - 1) / 2, 0)
        I = I / N

        fig, ax = plt.subplots()
        ax.imshow(np.transpose((I / 2 + 0.5), (1, 0)), cmap='gray', origin='lower', extent = [-((N - 1) / 2), ((N - 1) / 2), -((N - 1) / 2), ((N - 1) / 2)])
        plt.show()
        plt.close(fig)

    # Create a series of periodic image (with random phases) and average them
    if int(args.current_step) == 4:
        N = int(args.N)
        I = np.zeros((N, N))

        for m in range(N):
            I = I + Periodic_basis(args, N, 0, m - (N - 1) / 2, random.randint(1, 100))
        I = I / N

        fig, ax = plt.subplots()
        ax.imshow(np.transpose((I / 2 + 0.5), (1, 0)), cmap='gray', origin='lower', extent = [-((N - 1) / 2), ((N - 1) / 2), -((N - 1) / 2), ((N - 1) / 2)])
        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convolution_and_Fourier_and_Filter")
    parser.add_argument('--path', default="data", type=str)
    parser.add_argument('--data', default="none", type=str)
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--N', default=31, type=int)
    parser.add_argument('--current_step', default=1, type=int)
    args = parser.parse_args()
    main(args)

    # Fill in the other students you collaborate with:
    # e.g., Wei-Lun Chao, chao209
    #
    #