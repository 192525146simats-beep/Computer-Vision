# Computer Vision & Railway Track Fault Detector

This repository contains a collection of OpenCV and Python computer vision experiments, along with an integrated application for automated railway track fault detection.

---

## 🚂 Railway Track Fault Detector

Located in the [`railway-fault-detector`](railway-fault-detector/) directory, this module provides an automated visual inspection system designed to analyze track integrity and detect defects using Computer Vision and Deep Learning.

### Overview & Key Features
- **Automated Inspection:** Analyzes real-time video streams and image feeds captured by inspection vehicles or drones, eliminating the need for manual track walking.
- **Defect Detection:** Identifies critical structural defects such as rail surface cracks, misalignments, damaged sleepers, missing fasteners/Pandrol clips, and obstacles on the track.
- **Real-Time Visual Alerts:** Overlays bounding boxes with defect labels and detection confidence scores directly on frame streams.
- **Web Interface:** Built on Flask (`app.py`), allowing operators to upload video feeds, monitor real-time detections, and review logs via a web browser.

---

## 📂 List of Experiments

1. **Exp 01:** Read an image using Python and convert it to Grayscale.
2. **Exp 02:** Read an image and apply Gaussian Blur.
3. **Exp 03:** Read an image and display the image outline using the Canny edge detector.
4. **Exp 04:** Implement Histogram Equalization and compare it with the original image using OpenCV.
5. **Exp 05:** Write a Python function to analyze the histogram of an input image based on color levels.
6. **Exp 06:** Perform basic image processing and convert the image into an eroded image using the Erode function.
7. **Exp 07:** Read a video using Python and display it in slow motion and fast motion.
8. **Exp 08:** Dilate an image using the Dilate function.
9. **Exp 09:** Implement image scaling techniques to resize images to larger and smaller sizes.
10. **Exp 10:** Perform 90-degree clockwise rotation of an image.
11. **Exp 11:** Perform 180-degree clockwise rotation of an image.
12. **Exp 12:** Perform 270-degree clockwise rotation of an image.
13. **Exp 13:** Perform Affine Transformation on an image using Python and OpenCV.
14. **Exp 14:** Perform Perspective Transformation on an image using Python and OpenCV.
15. **Exp 15:** Detect corners in an image using Harris Corner Detection.
16. **Exp 16:** Implement the Sobel algorithm using OpenCV to filter an input image.
17. **Exp 17:** Design and implement a watermarking technique to insert a watermark into an image using OpenCV.
18. **Exp 18:** Implement image cropping, copying, and pasting operations to select a Region of Interest (ROI).
19. **Exp 19:** Implement Erosion morphological operation using OpenCV.
20. **Exp 20:** Implement Dilation morphological operation using OpenCV.
21. **Exp 21:** Implement Opening morphological operation using OpenCV.
22. **Exp 22:** Implement Closing morphological operation using OpenCV.
23. **Exp 23:** Implement Top Hat morphological operation using OpenCV.
24. **Exp 24:** Implement Black Hat morphological operation using OpenCV.
25. **Exp 25:** Recognize a watch from an image using object recognition techniques.
26. **Exp 26:** Implement a function to reverse frames of a video and create a reverse video using OpenCV.
27. **Exp 27:** Implement a face detection algorithm to detect and locate human faces using OpenCV.
28. **Exp 28:** Implement a vehicle detection algorithm to detect vehicles in video frames using OpenCV.
29. **Exp 29:** Implement an eye detection algorithm using OpenCV.
30. **Exp 30:** Implement a smile detection algorithm using OpenCV.
31. **Exp 31:** Implement an image segmentation algorithm based on threshold values using OpenCV.
32. **Exp 32:** Create a white image of user-defined size with four colored boxes (Black, Blue, Green, Red) at each corner.
33. **Exp 33:** Draw a Rectangle shape using OpenCV.
34. **Exp 34:** Draw a Circle shape using OpenCV.
35. **Exp 35:** Display a user-entered text string on an image using OpenCV.
36. **Exp 36:** Perform background subtraction based on color levels using OpenCV.
37. **Exp 37:** Perform foreground subtraction based on color levels using OpenCV.
38. **Exp 38:** Count the number of faces in an input image using OpenCV.
39. **Exp 39:** Play a video in reverse mode with slow motion using OpenCV.
40. **Exp 40:** Extract text from videos using OpenCV and OCR techniques.
