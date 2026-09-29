import argparse
import os
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl


def data_loader(args):
    """
    Output:
        I: the image matrix of size N-by-M-by-3.
            The pixel values are within [0, 1].
            The first dimension is horizontal, left-right (N pixels).
            The second dimension is vertical, bottom-up (M pixels).
            The third dimension is color channels, from Red to Green to Blue.
    """

    if args.data in ["cds", "lighthouse"]:
        print("Using " + args.data + " photo")
        current_dir = os.getcwd()
        image_path = osp.join(current_dir, 'data', args.data + '.png')
        I = np.asarray(Image.open(image_path)).astype(np.float64)/255
        I = np.transpose(I, (1, 0, 2))
        I = I[:, ::-1, :3]

    elif args.data == "rectangle":
        print("Using rectangle shape")
        I = np.zeros((31, 31, 3))
        I[10:21, 10:21, :] = 1

    ## Display the input image
    fig, ax = plt.subplots()
    ax.imshow(np.transpose(I, (1, 0, 2)), origin='lower')
    ax.set_title('Input image: I')
    if args.display:
        plt.show()
    plt.close(fig)

    return I


def load_kernel(args):
    """
    Output:
        kernel: the 2D kernel matrix.
    """

    print("Use " + args.kernel + " kernel")

    if args.kernel == "average":
        kernel = np.ones((11,11)) / 121

    if args.kernel == "binomial":
        #### Your job 1.2 starts here: implement the 5-by-5 2D binomial filter (see textbook/lecture slides for details) ####
        kernel = np.zeros((5, 5)) # this is a placeholder, and you can remove this line completely

        kernel = kernel / np.sum(kernel) # keep this line to make your filter properly normalized
        #### Your job 1.2 ends here: implement the 5-by-5 2D binomial filter ####

    return kernel


def Convolution(args, I, kernel):
    """
    Input:
        I: input image, a 3D matrix of size N-by-M-by-3
        kernel: convolutional kernel, a 2D matrix of size (2R + 1)-by-(2R + 1)
    Output:
        I_out: convoluted image, a 3D matrix of size N-by-M-by-3
    TODO:
        You are to implement convolution. See the Homework slide deck for details.
    """

    ## Initiate the output
    if np.size(I.shape) != 3:
        I = I.reshape((I.shape[0], I.shape[1], 1))
    I_out = np.zeros(I.shape)

    #### Your job 1.0 starts here: zero padding (just read the code) ####
    R = int((kernel.shape[0]-1)/2)
    I_pad = np.concatenate((np.zeros((R, I.shape[1], 3)), I, np.zeros((R, I.shape[1], 3))), axis=0)
    I_pad = np.concatenate((np.zeros((I_pad.shape[0], R, 3)), I_pad, np.zeros((I_pad.shape[0], R, 3))), axis=1)
    #### Your job 1.0 ends here: zero padding (just read the code) ####

    #### Your job 1.1 starts here: convolution (you may add just one more line without for loops; the fewer for loops you have; the faster the code runs) ####
    for c in range(I_pad.shape[2]):
        for n in range(I_out.shape[0]):
            for m in range(I_out.shape[1]):

                # Please fill in I_out

    #### Your job 1.1 ends here: convolution ####

    ## Display the convolution output image I_out
    fig, ax = plt.subplots()
    ax.imshow(np.transpose(I_out, (1, 0, 2)), origin='lower')
    ax.set_title('Convolution_output')
    if args.save:
        current_dir = os.getcwd()
        plt.savefig(osp.join(current_dir, 'result', str(
            args.current_step) + '_' + 'Convolution_output_' + args.data + '_' + args.kernel + '.png'))
        np.savez(osp.join(current_dir, 'result', str(
            args.current_step) + '_' + 'Results_Convolution_output_' + args.data + '_' + args.kernel + '.npz'),
                 m=I_out)
    if args.display:
        plt.show()

    plt.close(fig)

    return I_out


def Downsampling(args, I, kernel):
    """
    Input:
        I: input image, a 3D matrix of size N-by-M-by-3
        kernel: convolutional kernel, a 2D matrix of size (2R + 1)-by-(2R + 1)
    Output:
        I_out: downsampled image, a 3D matrix of size N/2-by-M/2-by-3
    """

    #### Your job 2 starts here: downsampling ####


    #### Your job 2 ends here: downsampling ####

    return I_out


def Upsampling(args, I, kernel):
    """
    Input:
        I: input image, a 3D matrix of size N-by-M-by-3
        kernel: convolutional kernel, a 2D matrix of size (2R + 1)-by-(2R + 1)
    Output:
        I_out: downsampled image, a 3D matrix of size 2N-by-2M-by-3
    """

    ## Initiate the output
    I_out = np.zeros((int(I.shape[0] * 2), int(I.shape[1] * 2), int(I.shape[2])))

    #### Your job 4 starts here: upsampling ####


    #### Your job 4 ends here: upsampling ####

    return I_out


def Gaussian_pyramid(args, I, kernel, scale):
    """
    Input:
        I: input image, a 3D matrix of size N-by-M-by-3
        kernel: convolutional kernel, a 2D matrix of size (2R + 1)-by-(2R + 1)
        scale: number of times to downsample or upsample
    Output:
        Output: a dictionary, where Output[n] records the image after the n-th downsampling (n starts from 0 and ends at scale)
                Output[0] = I, the original image
    """

    ## Initiate the output
    Output = {}
    Output[0] = I

    print("Perform downsampling")
    for n in range(scale):
    #### Your job 3 starts here: create the n-th downsampled image ####


    #### Your job 3 starts here: create the n-th downsampled image ####

    ## Display the sequence of downsampled images
    fig, ax = plt.subplots(nrows = 1, ncols = scale + 1)
    for n in range(scale + 1):
        background = np.zeros(I.shape)
        background[0: Output[n].shape[0], 0: Output[n].shape[1], :] = Output[n]
        ax[n].imshow(np.transpose(background, (1, 0, 2)), origin='lower', interpolation='none', aspect='equal')
        ax[n].set_title('Downsampled_' + str(n))
        if args.save:
            current_dir = os.getcwd()
            np.savez(osp.join(current_dir, 'result', 'Downsampled_' + args.data + '_scale_' + str(n) + '.npz'), m=Output[n])
    if args.save:
        current_dir = os.getcwd()
        plt.savefig(osp.join(current_dir, 'result', 'Downsampled_' + args.data + '.png'))
    if args.display:
        plt.show()
    plt.close(fig)

    return Output


def Laplacian_pyramid(args, G_pyramid, kernel, scale):
    """
    Input:
        scale: number of times to downsample or upsample
        G_pyramid: a dictionary of `scale + 1` images. G_pyramid[0] is the original image, a 3D matrix of size N-by-M-by-3
        kernel: convolutional kernel, a 2D matrix of size (2R + 1)-by-(2R + 1)
    Output:
        Output: a dictionary, where Output[n] records the residual image between the n-th and (n+1)-th downsampled images
                (n starts from 0 and ends at scale - 1)
    """

    ## Initiate the output
    Output = {}

    print("Computing the residual")
    for n in range(scale):
    #### Your job 5 starts here: create the n-th residual image ####


    #### Your job 5 starts here: create the n-th residual image ####

    ## Display the sequence of residual images
    fig, ax = plt.subplots(nrows = 1, ncols = scale)
    for n in range(scale):
        background = np.zeros(G_pyramid[0].shape)
        background[0: Output[n].shape[0], 0: Output[n].shape[1], :] = Output[n] + 0.5
        ax[n].imshow(np.transpose(background, (1, 0, 2)), origin='lower', interpolation='none', aspect='equal')
        ax[n].set_title('Residual_' + str(n))
        if args.save:
            current_dir = os.getcwd()
            np.savez(osp.join(current_dir, 'result', 'Residual_' + args.data + '_scale_' + str(n) + '.npz'), m=Output[n])
    if args.save:
        current_dir = os.getcwd()
        plt.savefig(osp.join(current_dir, 'result', 'Residual_' + args.data + '.png'))
    if args.display:
        plt.show()
    plt.close(fig)

    return Output


def Image_reconstruction(args, I_small, L_pyramid, kernel, scale):
    """
    Input:
        scale: number of times to downsample or upsample
        I_small: the image after downsampling `scale` times, a 3D matrix of size N/(2^scale)-by-M/(2^scale)-by-3
        L_pyramid: a dictionary of `scale` residual images. L_pyramid[n] records the residual image between the n-th and (n+1)-th downsampled images
        kernel: convolutional kernel, a 2D matrix of size (2R + 1)-by-(2R + 1)
    Output:
        Output: the reconstructed image, a 3D matrix of size N-by-M-by-3
    """

    print("Perform upsampling")
    Output = I_small
    for n in range(scale):
    #### Your job 6 starts here: reconstruct the original image ####


    #### Your job 6 starts here: reconstruct the original image ####

    return Output


def main(args):

    ## Convolution
    if int(args.current_step) == 1:
        print("Load image")
        I = data_loader(args)
        print("Perform convolution")
        kernel = load_kernel(args)
        I_out = Convolution(args, I, kernel)

    # Number of times to downsample or upsample
    scale = int(args.scale)

    ## Create Gaussian Pyramid
    if int(args.current_step) >= 2:
        print("Load image")
        I = data_loader(args)
        kernel = load_kernel(args)
        G_pyramid = Gaussian_pyramid(args, I, kernel, scale)

    ## Create Laplacian Pyramid
    if int(args.current_step) >= 3:
        L_pyramid = Laplacian_pyramid(args, G_pyramid, kernel, scale)

    ## Reconstruct the original image
    if int(args.current_step) >= 4:
        I_reconstruct = Image_reconstruction(args, G_pyramid[scale], L_pyramid, kernel, scale)

        ## Sanity check
        if np.sum((I - I_reconstruct) ** 2) > (10 ** (-12)): # Check if your reconstruction is good enough
            print("Wrong! You can do better!")
        else:
            print("Your result looks good!")

        ## Display the original and reconstructed images
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(np.transpose(I, (1, 0, 2)), origin='lower', interpolation='none', aspect='equal')
        ax[0].set_title('Original')
        ax[1].imshow(np.transpose(I_reconstruct, (1, 0, 2)), origin='lower', interpolation='none', aspect='equal')
        ax[1].set_title('Reconstructed')
        if args.save:
            current_dir = os.getcwd()
            plt.savefig(osp.join(current_dir, 'result', 'Reconstruct_' + args.data + '.png'))
            np.savez(osp.join(current_dir, 'result', 'Reconstruct_' + args.data + '.npz'), m = I_reconstruct)
        if args.display:
            plt.show()
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convolution_and_Fourier_and_Filter")
    parser.add_argument('--path', default="data", type=str)
    parser.add_argument('--data', default="none", type=str)
    parser.add_argument('--kernel', default="binomial", type=str)
    parser.add_argument('--scale', default=3, type=int)
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--current_step', default=1, type=int)
    args = parser.parse_args()
    main(args)

    # Fill in the other students you collaborate with (name and username):
    # e.g., Wei-Lun Chao, chao209
    #
    #