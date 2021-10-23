## Lane and Stopline Detection for Urban Roads <br>
The repository contains code for detecting lanes and stoplines on urban roads. It uses image processing based techniques to extract lane features, suppress noise and fit a parabolic lane model to detect urban lane markings. For stopline detection, clustering and principal component analysis based techinques are used to identify horizontal markings on roads. The code has been tested on the KITTI dataset, and has been deployed on the Mahindra E20 autonomous vehicle. <br>
<br>
The module is integrated on the ROS platform, and uses the [Dynamic Reconfigure](http://wiki.ros.org/dynamic_reconfigure) package to tune parameters. Description for various tuning parameters and working values have been provided in the code. The module uses input from an RGB camera published on `/camera/image_color`, and outputs the lane markings on `/lane_points`, a polygon geometry message.  

### Video Link: 
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/0gNgN58NdnY/0.jpg)](https://www.youtube.com/watch?v=0gNgN58NdnY)

