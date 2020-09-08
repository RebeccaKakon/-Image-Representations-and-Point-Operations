
our assignment includ a few function about Image Representations and Point Operations
1.
imReadAndConvert- a function that reads a given image file and converts it into a given representation. if the representation is 1 we convert the imag 
to grey scal and we craet a function rgb2gray that help us to do the right convertion. and if the representation =2 we convert to RGB .
2.
imDisplay
this function utilizes imReadAndConvert to display a given image file in a given representation.


The function is opening a new figure window and display the loaded image in the converted representation.
3.
transformRGB2YIQ- 
this function transform an RGB image into the YIQ . Given the red (R), green (G), and blue (B) pixel components of an RGB color image,
the corresponding luminance (Y), 
and the chromaticity components (I and Q) in the YIQ color space are
linearly related
we add a function get_YIQ_trans to to help us with the  the calculation of the function (just to have more order in the code) 
4.
transformRGB2YIQ - the opposite of transformRGB2YIQ. also here we used the get_YIQ_trans function. Because its the opposite we took the transpose matrix.
5.
hsitogramEqualize
this function performs histogram equalization of a given grayscale or RGB image. The function 
also display the input and the equalized output image
we used a functiom from numpy: interp, she upllayed all the information we collect to the new y channel we want to give back, by the logic we need (simuler to the lut function)
we add a function isGrayScale that return true if it is GrayScale and false if it is not !
6.
quantizeImage
This function performs optimal quantization of a given grayscale or RGB image. The function
return:

• A list of the quantized image in each iteration

• A list of the MSE error in each iteration
we add a function initialize_zq , that initializes z, q arrays with about equal amount of px in each bin
we use our function isGrayScale that return true if it is GrayScale and false if it is not.
we add a function optimize_z_q to chake our optimize z,q
after this we use our function apply_quantization that every Iteration we insert to our list of image the imege after the apply quantization, and we insert the error 
that we calculated by the mse formula
7.
gammaDisplay
This function performs gamma correction on an image with a given ?(that ?  between 0 to 2]
can woke on a grey scal and rbg .

