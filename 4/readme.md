# lego-searching

The code was made in MATLAB to count the number of blue rectangles (numA) and number of red squares (numB) on training images.

To initialize the code:

1. Download the repository lego-searching
2. Open initialize.m in MATLAB and select which training image to process, i.e. train__.jpg
3. Run the script

### PROCEDURE:

1. Download image as I=imread('filename') and invoke lego_count(I) function
2. Use K-Means Clustering to segment the image into different colors
3. Determine which clusters contain blue and red colors
4. Select blue and red clusters containing the most number of blue and red elements, respectively
5. Edge detection using Canny edge detector for red and blue colors
6. Dilation of objects and filling holes
7. Apply watershed method to separate touching objects and determine boundaries of objects
8. Compute properties of each identified object (areas, perimeters, etc.)
9. Classify each object as square/rectangle/circle/triangle
10. If a blue object is rectangular or red object is square, then we can judge whether an object is blue rectangle or red square based on their perimeters and area properties
11. Count the objects that fall under those categories
### REFERENCES:

Title: Matlab: Color-Based Segmentation Author: user2916044 Date: 5 September 2014 Availability: https://stackoverflow.com/questions/25691735/matlab-color-based-segmentation

Title: Matlab - How to detect green color on image? Author: drorco Date: 7 June 2016 Availability: https://stackoverflow.com/questions/37684903/matlab-how-to-detect-green-color-on-image

Title: Watershed transform question from tech support Author: Steve Eddins Date: 19 November 2013 Availability: https://blogs.mathworks.com/steve/2013/11/19/watershed-transform-question-from-tech-support/

Title: Detecting a Cell Using Image Segmentation Author: N/A Date: N/A Availability: https://www.mathworks.com/help/images/detecting-a-cell-using-image-segmentation.html

Title: Calculating perimeter of object Author: Image Analyst Date: 22 October 2016 Availability: https://www.mathworks.com/matlabcentral/answers/308183-calculating-perimeter-of-object

Title: How to classify shapes of this image as square, rectangle, triangle and circle Author: Matt Kindig Date: 21 February 2014 Availability: https://www.mathworks.com/matlabcentral/answers/116793-how-to-classify-shapes-of-this-image-as-square-rectangle-triangle-and-circle

Title: A suite of minimal bounding objects Author: John D'Errico Date: 23 May 2014 Availability: https://www.mathworks.com/matlabcentral/fileexchange/34767-a-suite-of-minimal-bounding-objects
