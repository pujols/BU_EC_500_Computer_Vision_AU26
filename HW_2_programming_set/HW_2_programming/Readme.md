# Homework 2

## Submission instructions

* Due date and time: October 14th (Monday) 2025, 23:59 ET

* Carmen submission: 
Submit a .zip file named `name.number.zip` (e.g., `chao.209.zip`) with the following files
  - your completed Python script `main.py`
  - your generated figures in the .png format (please see each question for what needs to be submitted) 
  - your generated files in the .npz format (please see each question for what needs to be submitted)

* Collaboration: You may discuss the homework with your classmates. However, you must write your solutions, complete your .py files, and submit them yourself. Collaboration in the sense that each of you completes some parts and then exchanges the solutions is NOT allowed. I do expect that your solutions won't be exactly the same. In your submission, you must list with whom you have discussed the homework. Please list each classmate's name and name.number (e.g., Wei-Lun Chao, chao.209) as a row at the end of `main.py`. That is, if you discussed your homework with two classmates, your .py file will have two rows at the end. Please consult the syllabus for what is and is not acceptable collaboration.

## Implementation instructions

* Download or clone this repository.

*  You will see a PPT and PDF named `HW2`, which provides useful information for the homework assignment.

* You will see one Python script: `main.py`.

* You will see a folder `for_display`, which contains some images used for display here.

* You will see a folder `data`, which contains some images used in the homework.

* You will see a folder `result`, which will save the generated results.

* Please use python3 and write your solutions from scratch. (You must use python3.)

* **Caution! Python and NumPy's indices start from 0. That is, to get the first element in a vector, the index is 0 rather than 1.**

* We note that the provided commands are designed to work with Mac/Linux with Python version 3. If you use Windows (like me!), we recommend that you run the code in the Windows command line (CMD). You may use `py -3` instead of `python3` to run the code. You may use editors like PyCharm to write your code.

* **Caution! Please do not import packages (like scikit learn) that are not listed in the provided code. In this homework, you are not allowed to use numPy's or other Python libraries' built-in convolution, filter functions, down-sampling, up-sampling, Gaussian pyramid, and Laplacian pyramid functions. If you use them, you will get 0 points for the entire homework.** 

* Caution! Follow the instructions in each question strictly to code up your solutions. **Do not change the output format. Do not modify the code unless we instruct you to do so.** (You are free to play with the code, but your submitted code should not contain those changes that we do not ask you to do.) A homework solution that does not match the provided setup, such as format, name, initializations, etc., will not be graded. It is your responsibility to make sure that your code runs with the provided commands and scripts.

## Installation instructions

* You will be using [NumPy] (https://numpy.org/), and your code will display your results with [matplotlib] (https://matplotlib.org/). If your computer does not have them, you may install them with the following commands:
  - for NumPy: <br/>
    do `sudo apt install python3-pip` or `pip3 install numpy`. If you are using the Windows command line, you may try `setx PATH "%PATH%;C:\Python34\Scripts"`, followed by `py -3 -mpip install numpy`.

  - for matplotlib: <br/>
    do `python3 -m pip install -U pip` and then `python3 -m pip install -U matplotlib`. If you are using the Windows command line, you may try `py -3 -mpip install -U pip` and then `py -3 -mpip install -U matplotlib`.



# Introduction

* In this homework, you will implement convolution and one filter (a convolutional kernel) in Lectures 6 & 7 & 9 (textbook, chapters 15 & 17). 

* In this homework, you will implement down-sampling, up-sampling, the Gaussian pyramid, and the Laplacian pyramid introduced in the textbook chapter 23. 

* You are given several images in the `data` folder, as well as the following toy image. All of them have three color channels (red, green, and blue). The pixel values are between 0.0 to 1.0.

Rectangle: 

![Alt text](https://github.com/pujols/OSU_CSE_5524_2025AU/blob/main/HW_2_programming_set/HW_2_programming/for_display/rectangle.png)

* The convoluted rectangle with an average (box) filter is as below:

![Alt text](https://github.com/pujols/OSU_CSE_5524_2025AU/blob/main/HW_2_programming_set/HW_2_programming/for_display/Convolution_output_rectangle_average.png)



# Question 0: Get ready 

* Please overview `main.py`. It contains multiple sub-functions. Specifically, you may want to take a look at `data_loader`, `load_kernel`, `Convolution`, `Downsampling`, `Upsampling`, `Gaussian_pyramid`, `Laplacian_pyramid`, and `Image_reconstruction`.

* We note that a matrix and an image have different axis ordering and direction. In numPy, for a matrix `I`,  `I[i, j]` means the i-th row (top-down) and j-th column (left-right). In this homework, however, **please treat `I` and other matrices directly as images. That is, given `I`,  `I[i, j, :]` means the R, G, and B pixel values at the horizontal index i (left-right) and vertical index j (bottom-up). Namely, the color at the `(i, j)` pixel location.** Please note that i and j both start from 0.



# Question 1: Convolution and binomial filter (10 pts)

* Go to the `main` function and find `if int(args.current_step) == 1:`

* Given the input image `I`, you need to perform convolution of it using `kernel`. 

* We have implemented one kernel in the `load_kernel(args)` function.

* Your job is to complete the implementation of the `Convolution(args, I, kernel)` function and a 2D binomial kernel in the `load_kernel(args)` function. Please go to the function and carefully read the input, output, and instructions. You can assume that the actual inputs will follow the input format, and your goal is to generate the output numpy array `I_out`. Please make sure that your results follow the required numpy array shapes.

* You may search **`#### Your job 1.0`**, **`#### Your job 1.1`**, **`#### Your job 1.2`** to locate where to amend your implementation. You will see some instructions there. You are free to create more space in between.

* Caution! For this question, please follow the formula in `HW2.ppt` or `HW2.pdf`.

## Running and saving

* Once completed, please run the following commands<br/>
`python3 main.py --current_step 1 --data rectangle --kernel average --display --save`<br/>
`python3 main.py --current_step 1 --data dreese --kernel binomial --display --save`<br/>
These commands will run your code. You will see several generated images, and several texts displayed in command lines. 

* The code will generate `1_Convolution_output_rectangle_average.png`, `1_Convolution_output_dreese_binomial.png`, `1_Results_Convolution_output_dreese_binomial.npz`, and `1_Results_Convolution_output_rectangle_average.npz`, which you will include in your submission.
 
* **Running the code should only take several seconds. If it runs more than 20 seconds, you will get a 30% reduction for Questions 1.** 
 


# Question 2: Downsampling (10 pts)

* Go to the `main` function and find `if int(args.current_step) >= 2:` and read the corresponding code.

* We will use your implemented 2D binomial kernel in the `load_kernel(args)` function.

* Your job is to complete the implementation of the `Downsampling(args, I, kernel)` function. Please go to the function and carefully read the input, output, and instructions. You can assume that the actual inputs will follow the input format, and your goal is to generate the output numpy array `I_out`. Please make sure that your results follow the required numpy array shapes. 

* You may search **`#### Your job 2 starts here: downsampling ####`** to locate where to amend your implementation. You will see some instructions there. You are free to create more space in between.

* Caution! For this question, please follow the formula in `HW2.ppt` or `HW2.pdf`.



# Question 3: Gaussian pyramid (10 pts)

![Alt text](https://github.com/pujols/OSU_CSE_5524_2025AU/blob/main/HW_2_programming_set/HW_2_programming/for_display/Down_sampled.png)

* Go to the `main` function and find `if int(args.current_step) >= 2:` and read the corresponding code.

* Your job is to complete the implementation of the `Gaussian_pyramid(args, I, kernel, scale)` function. Please go to the function and carefully read the input, output, and instructions. You can assume that the actual inputs will follow the input format, and your goal is to generate the output numpy dictionary `Output`, corresponding to the downsampled images. Please make sure that your results follow the required numpy array shapes. 

* You may search **`#### Your job 3 starts here: create the n-th downsampled image ####`** to locate where to amend your implementation. You will see some instructions there. You are free to create more space in between.

* Caution! For this question, please follow the formula in `HW2.ppt` or `HW2.pdf`.

## Running and saving

* Once completed, please run the following command<br/>
`python3 main.py --current_step 2 --data lighthouse --display --save`<br/>
These commands will run your code. You will see several generated images, and several texts displayed in command lines. 

* The code will generate `Downsampled_lighthouse_scale_0.npz`, `Downsampled_lighthouse_scale_1.npz`, `Downsampled_lighthouse_scale_2.npz`, `Downsampled_lighthouse_scale_3.npz`, and `Downsampled_lighthouse.png`, which you will include in your submission.

* **Running the code should only take several seconds. If it runs more than 20 seconds, you will get a 30% reduction for Questions 2 and 3.** 



# Question 4: Upsampling (10 pts)

* Go to the `main` function and find `if int(args.current_step) >= 3:` and read the corresponding code.

* Your job is to complete the implementation of the `Upsampling(args, I, kernel)` function. Please go to the function and carefully read the input, output, and instructions. You can assume that the actual inputs will follow the input format, and your goal is to generate the output numpy array `I_out`. Please make sure that your results follow the required numpy array shapes. 

* You may search **`#### Your job 4 starts here: upsampling ####`** to locate where to amend your implementation. You will see some instructions there. You are free to create more space in between.

* Caution! For this question, please follow the formula in `HW2.ppt` or `HW2.pdf`.



# Question 5: Laplacian pyramid (10 pts)

![Alt text](https://github.com/pujols/OSU_CSE_5524_2025AU/blob/main/HW_2_programming_set/HW_2_programming/for_display/Up_sampled.png)

* Go to the `main` function and find `if int(args.current_step) >= 3:` and read the corresponding code.

* Your job is to complete the implementation of the `Laplacian_pyramid(args, G_pyramid, kernel, scale)` function. Please go to the function and carefully read the input, output, and instructions. You can assume that the actual inputs will follow the input format, and your goal is to generate the output numpy dictionary `Output`, corresponding to the residual images. Please make sure that your results follow the required numpy array shapes. 

* You may search **`#### Your job 5 starts here: create the n-th residual image ####`** to locate where to amend your implementation. You will see some instructions there. You are free to create more space in between.

* Caution! For this question, please follow the formula in `HW2.ppt` or `HW2.pdf`.
  
## Running and saving

* Once completed, please run the following command<br/>
`python3 main.py --current_step 3 --data lighthouse --display --save`<br/>
These commands will run your code. You will see several generated images, and several texts displayed in command lines. 

* The code will generate `Residual_lighthouse_scale_0.npz`, `Residual_lighthouse_scale_1.npz`, `Residual_lighthouse_scale_2.npz`, and `Residual_lighthouse.png`, which you will include in your submission.

* **Running the code should only take several seconds. If it runs more than 40 seconds, you will get a 30% reduction for Questions 4 and 5.** 



# Question 6: Image reconstruction (10 pts)

![Alt text](https://github.com/pujols/OSU_CSE_5524_2025AU/blob/main/HW_2_programming_set/HW_2_programming/for_display/Reconstruct.png)

* Go to the `main` function and find `if int(args.current_step) >= 4:` and read the corresponding code.

* Your job is to complete the implementation of the `Image_reconstruction(args, I_small, L_pyramid, kernel, scale)` function. Please go to the function and carefully read the input, output, and instructions. You can assume that the actual inputs will follow the input format, and your goal is to generate the output numpy array `Output`, corresponding to the reconstructed image. Please make sure that your results follow the required numpy array shapes. 

* You may search **`#### Your job 6 starts here: reconstruct the original image ####`** to locate where to amend your implementation. You will see some instructions there. You are free to create more space in between.

* Caution! For this question, please follow the formula in `HW2.ppt` or `HW2.pdf`.

## Running and saving

* Once completed, please run the following command<br/>
`python3 main.py --current_step 4 --data lighthouse --display --save`<br/>
These commands will run your code. You will see several generated images, and several texts displayed in command lines. 

* The code will generate `Reconstruct_lighthouse.npz` and `Reconstruct_lighthouse.png`, which you will include in your submission.
  
* **Running the code should only take several seconds. If it runs more than 60 seconds, you will get a 30% reduction for Question 6.**

  

# What to submit:

* Please see the beginning of the page. Please follow **Submission instructions** to submit a .zip file named name.number.zip (e.g., chao.209.zip). Failing to submit a single .zip file will not be graded.
