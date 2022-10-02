# structure from motion using incremental method
The repository contains the python to reconstruct the scene using 2d images with the help of incremental SFM.
The camera poses obtained in the subsequent frame are relative to the first frame. Steps involved in the process is given below,  
1. Used SIFT to extract the keypoints and descriptors from the image. stored these features for future references for faster processing.
2. used the brute force matcher to match the descriptors and store the matches ofr the first time for future references.
3. first 2 views are used for constructing the baseline poses.
4. create 3d points for the baseline poses using triangulation
5. Incrementally, find the camera poses relative to the baseline views using PnP projection
6. Traingulate the new 3d points using the newly obtained pose from the PnP projection.
7. Do bundle adjustment using least squares for every few frames
8. repeat the process until all the images are processed


Run the solution using following command  
python3 main.py "path/to/your/images"  
The structure of dataset should be arranged as follows  

Root dir/  
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; images/  
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  0000.jpg  
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  0001.jpg   
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;   ...   
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;   K.txt

 Use meshlab to view the pointcloud stored in the reconstructed