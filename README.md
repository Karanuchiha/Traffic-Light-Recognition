# Traffic-Light-Recognition

The solution I implemented contained a main() function and a processImage() function. The main function
invoked the processImage() function for each of the 14 images to locate and recognise the traffic light and
it’s state. The processImage() function involved the following steps:

1. Converting to grayscale and thresholding: On observing that all the traffic lights are primarily black
regions with coloured light in between, I converted my image to a grayscale image and applied
thresholding to the image to separate the dark objects in the image from the lighter ones. This gave
me a binary image with all the dark objects white coloured and the background black.

2. Opening and closing morphological operations: Since in many images, the traffic lights were
overlapping dark coloured objects or had various dark objects near them, I applied opening (erosion
followed by dilation) and closing (dilation followed by erosion) to the image which had the effect of
separating the required traffic lights from the surrounding objects.

3. Watershed Segmentation: There were still a lot of objects excluding the traffic lights, present as
the foreground in the image. So, I applied watershed segmentation to further send some objects
into the background while keeping the necessary traffic lights in the foreground. To achieve this,
the binary image was severely eroded first to obtain a foreground image. Then, the binary image
was severely dilated first and then thresholding was applied to obtain the background image. The
severe erosion led to pixels that had a high chance of being in the foreground surviving, while
others were sent to the background. Severe dilation created blobs of the foreground objects and
thresholding was able to send pixels other than those in these objects to the background. The
foreground and background images were added to give the marker image which had pixels with
one of 3 colour values. Each value represented a different label (foreground, background or
undetermined). Using the original image and the marker image, watershed segmentation was
applied.

4. Contour Detection: Thresholding was applied to the watershed segmented image to get a binary
image with white foreground pixels and black background pixels. The next step was to find contours
in the obtained image for which the findContours() function was used. Now, the required contours
(representing the traffic lights) had to be extracted from all the obtained contours. To do this, I
iterated through all the obtained contours and obtained the minimum area rectangle for each of
the contours using the minAreaRect() function. The area of the contours, with the area of the holes
subtracted was calculated using the formula given in the OpenCV book[1]. To filter the contours, I
checked the following things:
• Rectangularity: ratio of the area of the bounding rectangle and the contour area. Only
contours with rectangularity greater than 0.9 were considered.
• Aspect Ratio: The width to height ratio of the considered contours was less than 0.5.
• Area of the contour: Only contours having area greater than a certain limit were considered.
After filtering, the regions of interest were extracted from the image, containing the obtained
contours.

5. Traffic Light State Detection: After obtaining the regions of interest in the image, the state of the
traffic light had to be detected. To do this, I used advantage of the fact that all the possible traffic
light colours had a separate range of hue values they fell in. So, I used a function that created a Hue
and Saturation mask for each colour (Green, Red and Amber) according to certain minimum and
maximum Hue and Saturation values by converting the ROI image to the HSV colour space and
thresholding.
For each region of interest, for each colour, the mask was applied to the ROI and hence, if pixels
belonging to the respective colour existed within the region of the traffic light, they were made
white, while all other pixels were made black.
The findContours() function was applied to this binary region of interest image to get any lights of
the respective colour that were in the region. Iterating through the obtained contours with a couple
of area filters, if a contour was detected, the region of interest was classified as having a light of
that colour.
The following images show various parts of the process and the end image shows both green lights
correctly identified. The bounding box for the right-hand light turned out a bit smaller and so
couldn’t be classified as a correctly located light in the program.

6. Performance Metrics: Performance metrics are calculated in the getPerformanceMetrics() function,
which calculates the number of true positives, false positives and false negatives by applying the
given constraints. Using the ground truth, variables of type Rect are created for each of the lights.
Using these Rects and the Rects found from the images, representing the bounding boxes of the
detected lights, the overlapping area is found. If the detected object passes the filters, it’s state is
compared with that of the ground truth. If it’s the same, the number of correctly identified states is
increased.
