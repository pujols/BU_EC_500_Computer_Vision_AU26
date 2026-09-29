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
        I = I[:, ::-1, :]

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


def matrix_to_list(args, I, display = False):
    """
    Input:
        I: input image, a 3D matrix of size N-by-M-by-3
    Output:
        I_list: a (N * M)-by-5 numPy array; each row saves (n, m, R, G, B);
                (n, m) means the pixel coordinate, starting from (0, 0);
                (R, G, B) means the corresponding pixel value.
    TODO:
        Please implement the conversion of image representation. See the Homework slide deck for details.
    """

    ## Initiate the output
    if np.size(I.shape) != 3:
        I = I.reshape((I.shape[0], I.shape[1], 1))
    I_list = np.zeros((I.shape[0] * I.shape[1], 5))

    #### Your job 1 starts here: ####

    #### Your job 1 ends here: ####

    ## Display the point cloud
    if display:
        fig, ax = plt.subplots()
        ax.scatter(I_list[::5, 0], I_list[::5, 1], c = I_list[::5, 2:], s = 0.1)
        ax.set_title('Show colorful point cloud')
        # if args.save:
        #     current_dir = os.getcwd()
        #     plt.savefig(osp.join(current_dir, 'result', str(
        #         args.current_step) + '_' + 'matrix_to_list_output_' + args.data + '.png'))
        #     np.savez(osp.join(current_dir, 'result', str(
        #         args.current_step) + '_' + 'Results_matrix_to_list_output_' + args.data + '.npz'),
        #              m=I_list)
        # if args.display:
        #     plt.show()
        plt.show()
        plt.close(fig)

    return I_list


def forward_mapping(args, I, transformation_mat):
    """
    Input:
        I: input image, a 3D matrix of size N-by-M-by-3
        transformation_mat: a 3-by-3 matrix converting the pixels coordinates from the input image to the output image
    Output:
        I_out: output image, a 3D matrix of size N-by-M-by-3
    TODO:
        Please implement the forward mapping method. See the Homework slide deck for details.
    """

    ## Initiate the output
    if np.size(I.shape) != 3:
        I = I.reshape((I.shape[0], I.shape[1], 1))
    I_out = np.zeros(I.shape)

    #### Your job 4 starts here: forward_mapping ####
    # Please use the function matrix_to_list


    #### Your job 4 ends here: forward_mapping ####

    ## Display the forward_mapping output image I_out
    fig, ax = plt.subplots()
    ax.imshow(np.transpose(I_out, (1, 0, 2)), origin='lower')
    ax.set_title('Forward_mapping_output')
    if args.save:
        current_dir = os.getcwd()
        plt.savefig(osp.join(current_dir, 'result', str(
            args.current_step) + '_' + 'forward_mapping_output_' + args.data + '_' + str(args.mat_ID) + '.png'))
        np.savez(osp.join(current_dir, 'result', str(
            args.current_step) + '_' + 'Results_forward_mapping_output_' + args.data + '_' + str(args.mat_ID) + '.npz'),
                 m=I_out)
    if args.display:
        plt.show()

    plt.close(fig)

    return I_out


def backward_mapping(args, I, transformation_mat):
    """
    Input:
        I: input image, a 3D matrix of size N-by-M-by-3
        transformation_mat: a 3-by-3 matrix converting the pixels coordinates from the input image to the output image
    Output:
        I_out: output image, a 3D matrix of size N-by-M-by-3
    TODO:
        Please implement the backward mapping method. See the Homework slide deck for details.
    """

    ## Initiate the output
    if np.size(I.shape) != 3:
        I = I.reshape((I.shape[0], I.shape[1], 1))
    I_out = np.zeros(I.shape)

    #### Your job 5 starts here: backward_mapping ####
    # Please use the function matrix_to_list


    #### Your job 5 ends here: backward_mapping ####

    ## Display the backward_mapping output image I_out
    fig, ax = plt.subplots()
    ax.imshow(np.transpose(I_out, (1, 0, 2)), origin='lower')
    ax.set_title('Backward_mapping_output')
    if args.save:
        current_dir = os.getcwd()
        plt.savefig(osp.join(current_dir, 'result', str(
            args.current_step) + '_' + 'backward_mapping_output_' + args.data + '_' + str(args.mat_ID) + '.png'))
        np.savez(osp.join(current_dir, 'result', str(
            args.current_step) + '_' + 'Results_backward_mapping_output_' + args.data + '_' + str(args.mat_ID) + '.npz'),
                 m=I_out)
    if args.display:
        plt.show()

    plt.close(fig)

    return I_out


def transformation_mat_1(args):
    """
    Output:
        transformation_mat: a 3-by-3 matrix converting the pixels coordinates from the input to the output image
    TODO:
        Please implement the following geometric operations sequentially. See the Homework slide deck for details.
        (1) translate each pixel to the right by 2; to the top by 5
        (2) rotate clockwise by 10 degrees
        (3) scale the horizontal direction by 0.8; vertical direction by 1.2
    """

    ## Initiate the output
    transformation_mat = np.zeros((3, 3))

    #### Your job 2 starts here ####

    #### Your job 2 starts here ####

    return transformation_mat


def transformation_mat_2(args):
    """
    Output:
        transformation_mat: a 3-by-3 matrix converting the pixels coordinates from the input to the output image
    TODO:
        Please implement the following geometric operations sequentially. See the Homework slide deck for details.
        (1) rotate the image counterclockwise "w.r.t. the coordinate (250, 250)" by 45 degrees
        (2) scale the horizontal direction by 1.6 and vertical direction by 1.2 "w.r.t. the coordinate (250, 250)"
    """

    ## Initiate the output
    transformation_mat = np.zeros((3, 3))

    #### Your job 3 starts here ####

    #### Your job 3 starts here ####

    return transformation_mat


def main(args):

    ## Convert the image matrices to a list
    if int(args.current_step) == 1:
        print("Load image")
        I = data_loader(args)
        print("Perform image matrices to a list conversion")
        I_list = matrix_to_list(args, I, True)

    ## Create transformation matrices
    if int(args.current_step) >= 2:
        transformation_list = {1: transformation_mat_1(args), 2 : transformation_mat_2(args)}

    ## Perform forward mapping
    if int(args.current_step) == 2:
        print("Load image for the forward mapping")
        I = data_loader(args)
        forward_mapping(args, I, transformation_list[int(args.mat_ID)])

        ## Perform backward mapping
    if int(args.current_step) == 3:
        print("Load image for the backward mapping")
        I = data_loader(args)
        backward_mapping(args, I, transformation_list[int(args.mat_ID)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convolution_and_Fourier_and_Filter")
    parser.add_argument('--path', default="data", type=str)
    parser.add_argument('--data', default="none", type=str)
    parser.add_argument('--mat_ID', default=2, type=int)
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--current_step', default=1, type=int)
    args = parser.parse_args()
    main(args)

    # Fill in the other students you collaborate with (name and username):
    # e.g., Wei-Lun Chao, chao209
    #
    #