import argparse
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl


def data_loader(args):
    """
    Output:
        I: a 2D matrix, i.e., image. Pixel values are between 0 and 1. The larger, the brighter.
    """

    I = np.ones((41, 41))
    I[5: 31, 10: 36] = 0.5
    I[5, 20: 36] = 0
    I[15, 10: 26] = 0
    I[30, 10: 26] = 0
    I[15: 31, 10] = 0
    I[15: 31, 25] = 0
    I[5:21, 35] = 0

    for i in range(9):
        I[6+i, 19-i] = 0
        I[6+i, 34-i] = 0
        I[21+i, 34-i] = 0

    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if i + j > 55:
                I[i, j] = 1
            elif i + j < 25:
                I[i, j] = 1

    ## Display
    fig, ax = plt.subplots()
    ax.imshow(I.transpose(), cmap='gray', origin='lower')
    ax.set_title('Simple_Vision_System: I')

    if args.save:
        plt.savefig('I.png')

    if args.display:
        plt.show()

    plt.close(fig)

    return I


def find_background(args, I):
    """
    Input:
        I: a 2D matrix, i.e., image
    Output:
        Map_background: a 0-1 map, with the same size as I
    TODO:
        If a pixel (i, j) in I belongs to background, save 1 in the (i, j) entry of Map_background; otherwise, 0
    """

    #### Your job 1 starts here: generate Map_background (< 5 lines) ####



    #### Your job 1 ends here: generate Map_background ####

    ## Display
    fig, ax = plt.subplots()
    ax.imshow(Map_background.transpose(), cmap='gray', origin='lower')
    ax.set_title('Map_background')

    if args.save:
        plt.savefig('Map_background.png')
        np.savez('Results_Map_background.npz', m = Map_background)

    if args.display:
        plt.show()

    plt.close(fig)

    return Map_background


def find_edge(args, I):
    """
    Input:
        I: a 2D matrix, i.e., image
    Output:
        Map_edge: a 0-1 map, with the same size as I
    TODO:
        If a pixel (i, j) in I belongs to edges, save 1 in the (i, j) entry of Map_edge; otherwise, 0
    """

    Map_edge = (I < 0.1).astype(float)

    ## Display
    fig, ax = plt.subplots()
    ax.imshow(Map_edge.transpose(), cmap='gray', origin='lower')
    ax.set_title('Map_edge')

    if args.save:
        plt.savefig('Map_edge.png')

    if args.display:
        plt.show()

    plt.close(fig)

    return Map_edge


def find_surface(args, I):
    """
    Input:
        I: a 2D matrix, i.e., image
    Output:
        Map_surface: a 0-1 map, with the same size as I
    TODO:
        If a pixel (i, j) in I belongs to surface, save 1 in the (i, j) entry of Map_surface; otherwise, 0
    """

    Map_surface = (I == 0.5).astype(float)

    ## Display
    fig, ax = plt.subplots()
    ax.imshow(Map_surface.transpose(), cmap='gray', origin='lower')
    ax.set_title('Map_surface')

    if args.save:
        plt.savefig('Map_surface.png')

    if args.display:
        plt.show()

    plt.close(fig)

    return Map_surface


def find_edge_type(args, I, Map_background, Map_edge, Map_surface):
    """
    Input:
        I: a 2D matrix, i.e., image
        Map_background: a 2D matrix of the same size as I; Map_background[i, j] = 1 if the pixel belongs to background; otherwise, 0
        Map_edge:       a 2D matrix of the same size as I; Map_edge[i, j] = 1 if the pixel belongs to edges; otherwise, 0
        Map_surface:    a 2D matrix of the same size as I; Map_surface[i, j] = 1 if the pixel belongs to surface; otherwise, 0
    Output:
        Map_horizontal_edge:    a 2D matrix of the same size as I; Map_horizontal_edge[i, j] = 1 if the pixel belongs to horizontal edges; otherwise, 0
        Map_vertical_edge:      a 2D matrix of the same size as I; Map_vertical_edge[i, j] = 1 if the pixel belongs to vertical edges; otherwise, 0
        Map_contact_edge:       a 2D matrix of the same size as I; Map_contact_edge[i, j] = 1 if the pixel belongs to contact edges; otherwise, 0
    TODO:
        Build the three edge maps for the three edge types
    """

    Map_horizontal_edge = np.zeros(I.shape)
    Map_vertical_edge = np.zeros(I.shape)
    Map_contact_edge = np.zeros(I.shape)

    #### Your job 2 starts here: generate Map_contact_edge (<10 lines) ####



    #### Your job 2 ends here: generate Map_contact_edge ####

    ## Compute the edge angle for each pixel (i, j)
    x_diff_down = np.concatenate((np.zeros((1, I.shape[1])), I[1:, :] - I[0:-1, :]), axis=0)
    y_diff_down = np.concatenate((np.zeros((I.shape[0], 1)), I[:, 1:] - I[:, 0:-1]), axis=1)
    x_diff_up = np.concatenate((I[0:-1, :] - I[1:, :], np.zeros((1, I.shape[1]))), axis=0)
    y_diff_up = np.concatenate((I[:, 0:-1] - I[:, 1:], np.zeros((I.shape[0], 1))), axis=1)
    edge_angle = np.abs(np.arctan((y_diff_down + 0.000001) / (x_diff_down + 0.000001))) + np.abs(np.arctan((y_diff_up + 0.000001) / (x_diff_up + 0.000001)))

    #### Your job 3 starts here: generate Map_horizontal_edge & Map_vertical_edge (<15 lines) ####



    #### Your job 3 ends here: generate Map_horizontal_edge & Map_vertical_edge ####

    ## Display
    fig, ax = plt.subplots()
    ax.imshow(Map_horizontal_edge.transpose(), cmap='gray', origin='lower')
    ax.set_title('Map_horizontal_edge')
    if args.save:
        plt.savefig('Map_horizontal_edge.png')
        np.savez('Results_Map_horizontal_edge.npz', m=Map_horizontal_edge)

    if args.display:
        plt.show()

    plt.close(fig)

    fig, ax = plt.subplots()
    ax.imshow(Map_vertical_edge.transpose(), cmap='gray', origin='lower')
    ax.set_title('Map_vertical_edge')

    if args.save:
        plt.savefig('Map_vertical_edge.png')
        np.savez('Results_Map_vertical_edge.npz', m=Map_vertical_edge)

    if args.display:
        plt.show()

    plt.close(fig)

    fig, ax = plt.subplots()
    ax.imshow(Map_contact_edge.transpose(), cmap='gray', origin='lower')
    ax.set_title('Map_contact_edge')

    if args.save:
        plt.savefig('Map_contact_edge.png')
        np.savez('Results_Map_contact_edge.npz', m=Map_contact_edge)

    if args.display:
        plt.show()

    plt.close(fig)

    return Map_horizontal_edge, Map_vertical_edge, Map_contact_edge


def image_plane(args, I):
    """
    Input:
        I: a 2D matrix, i.e., image
    Output:
        xx: a 2D matrix with the same size as I. xx[i, j] is the 2D x-value of the pixel (i, j) in I
        yy: a 2D matrix with the same size as I. yy[i, j] is the 2D y-value of the pixel (i, j) in I
    TODO:
    """

    xx = np.zeros(I.shape)
    yy = np.zeros(I.shape)

    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            xx[i, j] = i - int(I.shape[0] / 2)
            yy[i, j] = j - int(I.shape[1] / 2)

    ## Display
    fig, ax = plt.subplots()
    ax.imshow(xx.transpose(), cmap='gray', origin='lower')
    ax.set_title('2D_x')

    if args.save:
        plt.savefig('2D_x.png')

    if args.display:
        plt.show()

    plt.close(fig)

    fig, ax = plt.subplots()
    ax.imshow(yy.transpose(), cmap='gray', origin='lower')
    ax.set_title('2D_y')

    if args.save:
        plt.savefig('2D_y.png')

    if args.display:
        plt.show()

    plt.close(fig)

    return xx, yy


def X_3D(args, I, xx):
    """
    Input:
        I: a 2D matrix, i.e., image
        xx: a 2D matrix with the same size as I. xx[i, j] is the 2D x-value of the pixel (i, j) in I
    Output:
        X: a 2D maxrix with the same size as I. X[i, j] is the 3D X-value of the pixel (i, j) in I
    """

    X = xx

    ## Display
    fig, ax = plt.subplots()
    ax.imshow(X.transpose(), cmap='gray', origin='lower')
    ax.set_title('3D_X')

    if args.save:
        plt.savefig('3D_X.png')

    if args.display:
        plt.show()

    plt.close(fig)

    return X


def Y_3D(I, Map_background, Map_horizontal_edge, Map_vertical_edge, Map_contact_edge, Map_surface, theta):
    """
    Input:
        I: a 2D matrix, i.e., image
        Map_background:         a 2D matrix of the same size as I; Map_background[i, j] = 1 if the pixel belongs to background; otherwise, 0
        Map_horizontal_edge:    a 2D matrix of the same size as I; Map_horizontal_edge[i, j] = 1 if the pixel belongs to horizontal edges; otherwise, 0
        Map_vertical_edge:      a 2D matrix of the same size as I; Map_vertical_edge[i, j] = 1 if the pixel belongs to vertical edges; otherwise, 0
        Map_contact_edge:       a 2D matrix of the same size as I; Map_contact_edge[i, j] = 1 if the pixel belongs to contact edges; otherwise, 0
        Map_surface:            a 2D matrix of the same size as I; Map_surface[i, j] = 1 if the pixel belongs to surfaces; otherwise, 0
        theta: viewing angle of the camera
    Output:
        Y: a 2D maxrix with the same size as I. Y[i, j] is the 3D Y-value of the pixel (i, j) in I
    TODO:
        Calculate the Y values based on solving a linear system
    """

    # Compute the nx and ny map; you can access the nx and ny for pixel (i, j) by nx[i, j] and ny[i, j]
    x_diff_down = np.concatenate((np.zeros((1, I.shape[1])), I[1:, :] - I[0:-1, :]), axis=0)
    y_diff_down = np.concatenate((np.zeros((I.shape[0], 1)), I[:, 1:] - I[:, 0:-1]), axis=1)
    L2_norm = (x_diff_down ** 2 + y_diff_down ** 2) ** 0.5
    nx = x_diff_down / (L2_norm + 0.0001)
    ny = y_diff_down / (L2_norm + 0.0001)

    #### Your job 4 starts here: generate Y (<100 lines) ####



    #### Your job 4 ends here: generate Y ####

    ## Display
    fig, ax = plt.subplots()
    ax.imshow(Y.transpose(), cmap='gray', origin='lower')
    ax.set_title('3D_Y')
    if args.save:
        plt.savefig('3D_Y.png')
        np.savez('Results_3D_Y.npz', m=Y)

    if args.display:
        plt.show()

    plt.close(fig)

    return Y


def Z_3D(I, yy, Y, theta):
    """
    Input:
        I: a 2D matrix, i.e., image
        Y: a 2D maxrix with the same size as I. Y[i, j] is the 3D Y-value of the pixel (i, j) in I
        yy: a 2D matrix with the same size as I. yy[i, j] means the 2D y-value of the pixel (i, j) in I
        theta: viewing angle of the camera
    Output:
        Z: a 2D maxrix with the same size as I. Z[i, j] is the 3D Z-value of the pixel (i, j) in I
    """

    #### Your job 5 starts here: generate Y (<10 lines) ####



    #### Your job 5 ends here: generate Y ####

    ## Display
    fig, ax = plt.subplots()
    ax.imshow(Z.transpose(), cmap='gray', origin='lower')
    ax.set_title('3D_Z')

    if args.save:
        plt.savefig('3D_Z.png')
        np.savez('Results_3D_Z.npz', m=Z)

    if args.display:
        plt.show()

    plt.close(fig)

    return Z


def plot_3D(I, X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X.reshape(-1), Z.reshape(-1), Y.reshape(-1), c= (1- I.reshape(-1)),  cmap=mpl.colormaps["Greys"])
    ax.set_title("3D point cloud")
    plt.axis('tight')
    plt.xticks([]), plt.yticks([])
    plt.show()
    plt.close(fig)
    return

## Main function
def main(args):

    ######## Notes ###########
    ## All matrices I, x, y, Map_edge, Map_surface, Map_background, Map_horizontal_edge, Map_vertical_edge, Map_contact_edge, X, Y, Z have the same size
    ##########################

    ## Load the image for the simple vision system: parallel projection; camera angle is theta
    print("Load image")
    I = data_loader(args)
    theta = 1/8 * np.pi

    ## Generate the image coordinate location for each pixel
    print("Generate the image coordinate location for each pixel")
    x, y = image_plane(args, I)

    ## Generate the edge map
    print("Generate the edge map")
    Map_edge = find_edge(args, I)

    ## Generate the surface map
    print("Generate the surface map")
    Map_surface = find_surface(args, I)

    ## Generate the background map
    print("Generate the background map")
    if int(args.current_step) >= 1:
        Map_background = find_background(args, I)

    ## Generate the edge maps for horizontal edges, vertical edges, and contact edges
    print("Generate the edge maps for horizontal edges, vertical edges, and contact edges")
    if int(args.current_step) >= 3:
        Map_horizontal_edge, Map_vertical_edge, Map_contact_edge = find_edge_type(args, I, Map_background, Map_edge, Map_surface)

    ## Generate the 3D X coordinate location for each pixel
    print("Generate the 3D X coordinate location for each pixel")
    X = X_3D(args, I, x)

    ## Generate the 3D Y coordinate location for each pixel
    print("Generate the 3D Y coordinate location for each pixel")
    if int(args.current_step) >= 4:
        Y = Y_3D(I, Map_background, Map_horizontal_edge, Map_vertical_edge, Map_contact_edge, Map_surface, theta)

    ## Generate the 3D Z coordinate location for each pixel
    print("Generate the 3D Z coordinate location for each pixel")
    if int(args.current_step) >= 5:
        Z = Z_3D(I, y, Y, theta)

    ## Plot X, Y, Z in 3D
        print("Plot X, Y, Z in 3D as a point cloud")
        plot_3D(I, X, Y, Z)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple Vision System")
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--current_step', default=0, type=int)
    args = parser.parse_args()
    main(args)

    # Fill in the other students you collaborate with:
    # e.g., Wei-Lun Chao, chao209
    #
    #