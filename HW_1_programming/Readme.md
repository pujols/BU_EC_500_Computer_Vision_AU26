# Homework 1

## Submission instructions

* Due date and time: September 18th (Thursday) 2025, 23:59 ET

* Carmen submission: 
Submit a .zip file named `name.number.zip` (e.g., `chao.209.zip`) with the following files
  - your completed Python script `main.py`
  - your 6 generated figures `Map_background.png`, `Map_horizontal_edge.png`, `Map_vertical_edge.png`, `Map_contact_edge.png`, `3D_Y.png`, `3D_Z.png`
  - your 6 generated files `Results_Map_background.npz`, `Results_Map_horizontal_edge.npz`, `Results_Map_vertical_edge.npz`, `Results_Map_contact_edge.npz`, `Results_3D_Y.npz`, `Results_3D_Z.npz`.

* Collaboration: You may discuss the homework with your classmates. However, you need to write your solutions, complete your .py files, and submit them by yourself. In your submission, you must list with whom you have discussed the homework. Please list each classmate’s name and name.number (e.g., Wei-Lun Chao, chao.209) as a row at the end of `main.py`. That is, if you discussed your homework with two classmates, your .py file will have two rows at the end. Please consult the syllabus for what is and is not acceptable collaboration.

## Implementation instructions

* Download or clone this repository.

*  You will see a PPT and PDF named `HW1`, which provides useful information for the homework assignment.

* You will see two Python scripts: `main.py` and `numpy_example.py`.

* You will see a folder `for_display`, which contains some images used for display here.

* Please use python3 and write your solutions from scratch. (You must use python3.)

* **Caution! Python and NumPy's indices start from 0. That is, to get the first element in a vector, the index is 0 rather than 1.**

* We note that the provided commands are designed to work with Mac/Linux with Python version 3. If you use Windows (like me!), we recommend that you run the code in the Windows command line (CMD). You may use `py -3` instead of `python3` to run the code. You may use editors like PyCharm to write your code.

* Caution! Please do not import packages (like scikit learn) that are not listed in the provided code. Follow the instructions in each question strictly to code up your solutions. Do not change the output format. Do not modify the code unless we instruct you to do so. (You are free to play with the code, but your submitted code should not contain those changes that we do not ask you to do.) A homework solution that does not match the provided setup, such as format, name, initializations, etc., will not be graded. It is your responsibility to make sure that your code runs with the provided commands and scripts.

## Installation instructions

* You will be using [NumPy] (https://numpy.org/), and your code will display your results with [matplotlib] (https://matplotlib.org/). If your computer does not have them, you may install them with the following commands:
  - for NumPy: <br/>
    do `sudo apt install python3-pip` or `pip3 install numpy`. If you are using the Windows command line, you may try `setx PATH "%PATH%;C:\Python34\Scripts"`, followed by `py -3 -mpip install numpy`.

  - for matplotlib: <br/>
    do `python3 -m pip install -U pip` and then `python3 -m pip install -U matplotlib`. If you are using the Windows command line, you may try `py -3 -mpip install -U pip` and then `py -3 -mpip install -U matplotlib`.



# Introduction

In this homework, you will implement a simplified version of the simple version system introduced in Lectures 2 and 3 (textbook, chapter 2). Specifically, your code will output several maps, including the 3D Y and 3D Z maps.

* Specifically, you are given the following gray-scale image I (a 2D matrix) captured by parallel projection with a viewing angle of theta = 45 degrees. White color means a pixel value of 1; Gray color means a pixel value of 0.5; black color means a pixel value of 0.0.

![Alt text](https://github.com/pujols/OSU_CSE_5524_2025AU/blob/main/HW_1_programming_set/HW_1_programming/for_display/I.png)

* Your goal is to derive the 3D locations of each pixel. Specifically, you are tasked to derive the 3D Y (height) map and 3D Z (depth) map for each pixel.

![Alt text](https://github.com/pujols/OSU_CSE_5524_2025AU/blob/main/HW_1_programming_set/HW_1_programming/for_display/3D_Y.png)

![Alt text](https://github.com/pujols/OSU_CSE_5524_2025AU/blob/main/HW_1_programming_set/HW_1_programming/for_display/3D_Z.png)

As you can see, the Y and Z maps have the same size as the input image I. For each pixel location, the input image I records its color or light intensity, while the Y and Z maps record their height and depth in 3D.

# Notes before you start or while you are implementing

1. In your implementation, you should not hardcode the edge locations by reading how we generate the image I in the data_loader function. Instead, you should write an algorithm that relies on 1) each edge pixel's 2D angle to determine whether it is a vertical edge or a horizontal edge; and 2) each pixel's vertically adjacent pixels to determine whether it is a contact edge or not. The definitions of these edge categories can be found in the lecture slides or the textbook. If you simply hardcode the locations, you will earn 0 points for the corresponding question. ===> The TA and I will generate a different image I to run your code. 

2. It is OK that you assume the background color, surface color, and edge colors are fixed. That is, you can set a pixel value threshold to determine which pixels belong to the background.

3. It is fine if your resulting maps look a bit different from what we provided (like a few pixel differences). When we grade your homework, we will set a tolerance in comparing your results to ours.


# Question -1: NumPy Exercise

* You will use [NumPy] (https://numpy.org/) extensively in this homework. NumPy is a library for the Python programming language, which adds support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. NumPy has many great functions and operations that will make your implementation much easier. 

* If you are not familiar with Numpy, we recommend that you read this [tutorial] (https://cs231n.github.io/python-numpy-tutorial/) or other tutorials online and then play with some code to become familiar with it.

* We have provided some useful Numpy operations that you may want to use in `numpy_example.py`. You may want to comment out all the lines first and execute them one by one or in a group to see the results and the differences. You can run the command `python3 numpy_example.py`.

* Caution! Python and NumPy's indices start from 0. That is, to get the first element in a vector, the index is 0 rather than 1.



# Question 0: Get ready 

* Please overview `main.py`. It contains multiple sub-functions. The outputs are certain maps like edge or background maps, which have the same size as the input image I.

* We note that a matrix and an image have different axis ordering and direction. In Python, for a matrix `I`,  `I[i, j]` means the i-th row (top-down) and j-th column (left-right). In this question, however, **please treat `I` and other maps directly as images. That is, given `I`,  `I[i, j]` means the color at the horizontal index i (left-right) and vertical index j (bottom-up). Namely, the color at the `(i, j)` pixel location.** Please note that i and j both start from 0.



# Question 1:  (10 pts)

* Given the input image `I`, we have implemented the edge map  `Map_edge = find_edge(args, I)` and the surface `Map_surface = find_surface(args, I)` for you. A value of 1 means the corresponding pixel belongs to edges or surfaces, respectively.
  
* You are asked to complete the function `def find_background(args, I)`, which generate the background map. Please go to the function and carefully read the input, output, and instructions. You can assume that the actual inputs will follow the input format, and your goal is to generate the output numpy array `Map_background`: a value of 1 means the corresponding pixel belongs to the backgrounds. Please make sure that your results follow the required numpy array shapes. 

* You may search **`#### Your job 1`** to locate where to amend your implementation. You will see some instructions there. You are free to create more space in between.

## Running and saving

* Once completed, please run the following command<br/>
`python3 main.py --current_step 1 --display --save`<br/>
This command will run your code. You will see the resulting `Map_background` displayed in your command line. 

* The code will generate `Map_background.png` and  `Results_Map_background.npz`, which you will include in your submission.

* Your result should look like:

![Alt text](https://github.com/pujols/OSU_CSE_5524_2025AU/blob/main/HW_1_programming_set/HW_1_programming/for_display/Map_background.png)



# Question 2 & 3:  (10 + 20 pts)

* Given the input image `I` and the edge map  `Map_edge`, you are asked to complete the function `def find_edge_type(args, I, Map_background, Map_edge, Map_surface)`, which generates the maps for horizontal, vertical, and contact edges.

* Please go to the function and carefully read the input, output, and instructions. You can assume that the actual inputs will follow the input format, and your goal is to generate the three output numpy arrays `Map_horizontal_edge`, `Map_vertical_edge`, and `Map_contact_edge`: a value of 1 means a pixel belongs to the corresponding type of edges. Please make sure that your results follow the required numpy array shapes. 

* You may search **`#### Your job 2`** and **`#### Your job 3`** to locate where to amend your implementation. You will see some instructions there. You are free to create more space in between.

* **Hint:** Since we already provided `Map_edge`, which locates all the pixels belonging to edges, `Map_horizontal_edge[i, j]`, `Map_vertical_edge[i, j]`, and `Map_contact_edge[i, j]` can be 1 only when `Map_edge[i, j]` is 1.

## Running and saving

* Once completed, please run the following command<br/>
`python3 main.py --current_step 3 --display --save`<br/>
This command will run your code. You will see the resulting `Map_horizontal_edge`, `Map_vertical_edge`, and `Map_contact_edge` displayed in your command line. 

* The code will generate `Map_horizontal_edge.png`, `Map_vertical_edge.png`, `Map_contact_edge.png`, `Results_Map_horizontal_edge.npz`, `Results_Map_vertical_edge.npz`, and `Results_Map_contact_edge.npz`, which you will include in your submission.

* Your result should look like:

![Alt text](https://github.com/pujols/OSU_CSE_5524_2025AU/blob/main/HW_1_programming_set/HW_1_programming/for_display/Map_horizontal_edge.png)

![Alt text](https://github.com/pujols/OSU_CSE_5524_2025AU/blob/main/HW_1_programming_set/HW_1_programming/for_display/Map_vertical_edge.png)

![Alt text](https://github.com/pujols/OSU_CSE_5524_2025AU/blob/main/HW_1_programming_set/HW_1_programming/for_display/Map_contact_edge.png)



# Question 4: (40 pts)

* Given the input image `I`, the background map `Map_background`, surface map `Map_surface`, edge maps for different types `Map_horizontal_edge`, `Map_vertical_edge`, `Map_contact_edge`, and the camera angle `theta`, you are asked to complete the function `def Y_3D(I, Map_background, Map_horizontal_edge, Map_vertical_edge, Map_contact_edge, Map_surface, theta)`, which generates the Y (height) map in 3D.

* Please go to the function and carefully read the input, output, and instructions. You can assume that the actual inputs will follow the input format, and your goal is to generate the output numpy array `Y`. Please make sure that your results follow the required numpy array shapes. 

* You may search **`#### Your job 4`** to locate where to amend your implementation. You will see some instructions there. You are free to create more space in between.

* **Hint:** You need to create a matrix `A` and a vector `b` to apply the linear algebra formula to obtain the answer. 

## Running and saving

* Once completed, please run the following command<br/>
`python3 main.py --current_step 4 --display --save`<br/>
This command will run your code. You will see the resulting `3D_Y` displayed in your command line. 

* The code will generate `3D_Y.png` and `Results_3D_Y.npz`, which you will include in your submission.



# Question 5: (10 pts)

* Given the input image `I`, `Y`, the camera angle `theta`, and each pixel's 2D vertical location `yy`, you are asked to complete the function `def Z_3D(I, yy, Y, theta)`, which generates the Z (depth) map in 3D.

* Please go to the function and carefully read the input, output, and instructions. You can assume that the actual inputs will follow the input format, and your goal is to generate the output numpy array `Z`. Please make sure that your results follow the required numpy array shapes. 

* You may search **`#### Your job 5`** to locate where to amend your implementation. You will see some instructions there. You are free to create more space in between.

## Running and saving

* Once completed, please run the following command<br/>
`python3 main.py --current_step 5 --display --save`<br/>
This command will run your code. You will see the resulting `3D_Z` displayed in your command line. 

* The code will generate `3D_Z.png` and `Results_3D_Z.npz`, which you will include in your submission.



# What to submit:

* Please see the beginning of the page. Please follow **Submission instructions** to submit a .zip file named name.number.zip (e.g., chao.209.zip). Failing to submit a single .zip file will not be graded.
