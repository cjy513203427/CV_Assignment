# CV_Assignment
 Assignment of Computer Vision in WS2020/21 in TU Darmstadt

ProblemX.py is my work.  
ProblemX_.py is work from my teammate. His github is https://github.com/HanG-94  
ProblemX_Loesung is the solution from tutor.  

## Assignment 1
### Problem 1
**Getting to know Python**  
Results  
![image](https://user-images.githubusercontent.com/38099452/114175009-c5769280-9939-11eb-8188-385d5551b7be.png)  
mirrored image:  
![image](https://user-images.githubusercontent.com/38099452/114175069-da532600-9939-11eb-8123-abf86b944282.png)  
![image](https://user-images.githubusercontent.com/38099452/114175119-ea6b0580-9939-11eb-8dde-9b3520225543.png)  
### Problem 2
**Bayer Interpolation**  
Results  
![image](https://user-images.githubusercontent.com/38099452/114175266-1b4b3a80-993a-11eb-9313-033d41c1d77d.png)  
Assemble image:  
![image](https://user-images.githubusercontent.com/38099452/114175297-269e6600-993a-11eb-9ad0-f3ab8b6a2ede.png)  
After interpolation:  
![image](https://user-images.githubusercontent.com/38099452/114175327-2e5e0a80-993a-11eb-8ba4-9bc9f21c9be7.png)  
### Problem 3
**Projective Transformation**  
Results  
![image](https://user-images.githubusercontent.com/38099452/114176148-0fac4380-993b-11eb-8148-1ede842b57ae.png)  
![image](https://user-images.githubusercontent.com/38099452/114176168-15098e00-993b-11eb-897b-1e6cef6c7f62.png)  
### Problem 4
**Edge Detection**  
Results  
Show filter results:  
![image](https://user-images.githubusercontent.com/38099452/114176825-f8ba2100-993b-11eb-8d95-864774523403.png)  
Show gradient magnitude:  
![image](https://user-images.githubusercontent.com/38099452/114176953-18e9e000-993c-11eb-8390-8f0c618e5922.png)  
threshold derivative:  
![image](https://user-images.githubusercontent.com/38099452/114177008-2dc67380-993c-11eb-8966-bda5bb336a58.png)  
Non maximum suppression:  
![image](https://user-images.githubusercontent.com/38099452/114177073-420a7080-993c-11eb-9220-cf98d7e63c4d.png)  
## Assignment 2
### Problem 1
**Build image pyramid**  
Results  
load image and build Gaussian pyramid:  
![image](https://user-images.githubusercontent.com/38099452/114177661-f86e5580-993c-11eb-803a-796cee0eb1b0.png)  
build Laplacian pyramid from Gaussian pyramid and amplifiy high frequencies of Laplacian pyramid:  
![image](https://user-images.githubusercontent.com/38099452/114177900-3d928780-993d-11eb-8228-30c555202928.png)  
reconstruct sharpened image from amplified Laplacian pyramid:  
![image](https://user-images.githubusercontent.com/38099452/114177932-48e5b300-993d-11eb-829e-b83f97073a56.png)  
### Problem 2
**PCA**  
Results  
Using 2 random images for testing:  
![image](https://user-images.githubusercontent.com/38099452/114178649-4e8fc880-993e-11eb-9d11-278ba9be31c7.png)  
Compute PCA reconstruction:  
![image](https://user-images.githubusercontent.com/38099452/114178740-68311000-993e-11eb-8ffe-448db1c281d5.png)  
Image search:  
![image](https://user-images.githubusercontent.com/38099452/114178877-9151a080-993e-11eb-8dfa-91f2a9ec368b.png)  
Interpolation:  
![image](https://user-images.githubusercontent.com/38099452/114178977-b514e680-993e-11eb-8f2b-e23305045789.png)  
## Assignment 3
### Problem 1  
**Hessian Matrix**
Results  
Show components of Hessian matrix:  
![image](https://user-images.githubusercontent.com/38099452/114181187-7af91400-9941-11eb-9422-a80ca1145aac.png)  
Compute and show Hessian criterion:  
![image](https://user-images.githubusercontent.com/38099452/114181240-8e0be400-9941-11eb-91a0-eea79d599981.png)  
Show all interest points where criterion is greater than threshold:  
![image](https://user-images.githubusercontent.com/38099452/114181315-a67bfe80-9941-11eb-8848-a2f092449592.png)  
Apply non-maximum suppression and show remaining interest points:  
![image](https://user-images.githubusercontent.com/38099452/114181344-aed43980-9941-11eb-8a0b-5710a8172d02.png)  
### Problem 2
**RANSAC**
Results  
Find matching keypoints:  
![image](https://user-images.githubusercontent.com/38099452/114182092-8b5dbe80-9942-11eb-9178-cc0cbf2798ca.png)   
Compute homography matrix via ransac:  
![image](https://user-images.githubusercontent.com/38099452/114182157-9a447100-9942-11eb-9d25-ec017c0bd948.png)  
Recompute homography matrix based on inliers:  
![image](https://user-images.githubusercontent.com/38099452/114182212-a8928d00-9942-11eb-9351-0506268ee6b9.png)  
## Assignment 4
### Problem 1  
**Epipolar Geometry**
Results  
Find keypoints in the left and right image:  
![image](https://user-images.githubusercontent.com/38099452/114183287-cf9d8e80-9943-11eb-8514-f1c38cf159f9.png)  
Find epipolar lines in the left and right image:  
![image](https://user-images.githubusercontent.com/38099452/114183362-e5ab4f00-9943-11eb-91fc-c3a114510d8f.png)  
### Problem 2
**Optical Flow**
Results  
![image](https://user-images.githubusercontent.com/38099452/114184010-987bad00-9944-11eb-9475-2db1334feb0e.png)  
## Bonus Assignment
### Problem 1  
**Neural Networks**
Results  
![image](https://user-images.githubusercontent.com/38099452/114184167-ceb92c80-9944-11eb-85a7-9433bb8bb43f.png)
![image](https://user-images.githubusercontent.com/38099452/114184205-d8db2b00-9944-11eb-8523-dd1d5b3889fb.png)


