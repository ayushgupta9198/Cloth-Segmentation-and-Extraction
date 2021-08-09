# Cloth-Segmentation-and-Extraction

Goal: 

To visualize and createCloth_segmentation and Cloth_extraction images respectively..

Model brief:

As model description , it is based on image processing based computation where we take input image as a source and covered the maximum part as image have as a part of padding process , after padding area with borders and edges we make a prediction of that area based on the source image which is called the masking.
After masking we transform the values respectively and convert it into add weighted parameter which we will make it as a separate cloth.
To make it in colour we have use openCV colour conversion.

Vision towards goal:

Our mission is to generate uniques cloths based on  images for any custom image

Research involves around the model:

We implemented this model for custom images to visualize.
Model is available in both the versions like GPU & CPU.
Initially the model is available only on GPU functionality but I have edited the model code and made it working on custom images and works fine in both computations
We can change the image size dynamically to any resolution where it is working including texture generation , cloth generation , cloth mask generation and final output image generation.
	
Implementation towards model:

We have set up all the necessary things and models in progression of this model.
As you can check the model workflow ,  you just need to put the images in the right folder and code will generate results.
Please use the specific version of dependencies otherwise code will not work.
After run the cell one by one you can generate the final result either in current directory or in any specific folder.

Thanks :)
