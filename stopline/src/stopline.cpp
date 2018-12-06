#include "../include/stopline/stopline.hpp"

int isvalid(Mat A, int i, int j)
{
    if(i<A.rows && j< A.cols &&i>=0&&j>=0) return 1;
    return 0;
}

//function to calculate the eigenvector for each blob corresponding to eigenvalue of its dispersion matrix
float* PCa(vector<vector<Point> > data)
{
    //array of type CD
    float* theta_array;
    theta_array=(float*)malloc((data.size())*sizeof(CD));

    for(int i=0;i<data.size();i++)
    {
        float mean_x=0,mean_y=0;
        int k=data[i].size();
        MatrixXd cluster(2,5); 
        cluster.resize(2, k);
        //fill cluster with the x,y coordinates of each pixel in a blob
        for(int j=0;j<k;j++)
        {
            cluster(0 ,j)=data[i][j].x;
            cluster(1 ,j)=data[i][j].y;
            mean_x+=data[i][j].x;
            mean_y+=data[i][j].y;
        }
        mean_x=mean_x/k;
        mean_y=mean_y/k;
        //calculate the mean of x and y
        MatrixXd mean(2,2);
        mean << mean_x*mean_x , mean_x*mean_y,
                mean_y*mean_x , mean_y*mean_y;

        //find the dispersion matrix of the cluster using the formula DM=(cluster-mean)*(cluster-mean).transpose      
        MatrixXd dis_mat(2,2);
        dis_mat=(( cluster*(cluster.transpose()) )/(float)(k) ) - mean;
        //find out the eigenvalues and eigenvectors of dis_mat
        //eigenvalues are stored in a matrix es.eigenvalues() with 2x1 dimention and each entry is of type cv::Complex
        //similar storage of eigenvectors except they are stored in a 4x1 dimention matrix and each entry is complex
        EigenSolver<MatrixXd> es(dis_mat);

        double x,y;
        //store the eigenvector corresponding to the highest eigenvalue in the array theta_array
        if( real( es.eigenvalues()(0) )> real(es.eigenvalues()(1)) ) x= real(es.eigenvectors()(0)) , y= real(es.eigenvectors()(1)) ;
        else x= real(es.eigenvectors()(2)) , y= real(es.eigenvectors()(3)) ;
        //cout << es.eigenvalues() << endl << es. eigenvectors() << endl << x << " " << y << endl << endl;
        double theta;
        if( atan2(y,x)<0 ) theta=atan2(y,x)+3.14159265;
        else theta=atan2(y,x);
        theta=theta*(180/3.1415);
        (*(theta_array+i))= theta;
    }
    return theta_array;
}

void gradient_thresholding(Mat img, Mat dst)
{
    //pixel by pixel traversal for single channel grayscale image
   for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++) 
        {
            int value;
            Vec3b &intensity = img.at<Vec3b>(i, j);
            value = (2*intensity.val[0])-(intensity.val[1]); //Using 2B-G operation to enhance white lines
            
            if(value>155) //maintaining boundary conditions
                value=255;
            if(value<0) //maintaining boundary conditons
                value=0;

            if(value>160)
                dst.at<uchar>(i,j)=value; //changing the intensity value of dst_gray with enhanced values
                //cout<<"value ="<<(int)value<<"\t";
            if(value<160)
                dst.at<uchar>(i,j)=0;

        }
    }  
}

void gradient_thresholding_horizontal(Mat image,Mat original, Mat dst)//for lateral lane markings
{
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++) 
        {
            int grady,ity1,ity2,ity3;
            ity1=image.at<uchar>(i,j);
            
            if(isvalid(image,i-wi,j)==1 and isvalid(image,i+wi,j)==1) //preventing upper edge crossing by re-itterating
            {    
                ity2=image.at<uchar>(i-wi,j);
                ity3=image.at<uchar>(i+wi,j);
                //if(ity2==0) continue;
                grady=(2*ity1)-(ity2+ity3)+fabs(ity2-ity3); //gradient in y
                if(ity2==0||ity3==0) continue;
                if(grady>120 && original.at<Vec3b>(i,j)[0]>=150 &&original.at<Vec3b>(i,j)[1]>=150 && original.at<Vec3b>(i,j)[2]>=150 ) //manually set val for thresholding by comparing gradient value output
                    dst.at<uchar>(i,j)=255; 
            }

        }
    }
}

int colour(Mat tp,Mat dst_labels, int nlabels)
{

    //Random Colours vector definition DELETE FROM FINAL CODE
    vector<Vec3b> colors(nlabels);
    colors[0]= Vec3b(0,0,0);
    int i,j;
    for(i=1;i<nlabels;i++)
    {
        colors[i]=Vec3b((rand()%255),(rand()%255),(rand()%255));
    }
    //Assigning each blob a colour
    for(i=0;i<tp.rows;i++)
    {
        for(j=0;j<tp.cols;j++)
        {
            int label= dst_labels.at<int>(i,j);
            Vec3b &pixel=tp.at<Vec3b>(i,j);
            pixel=colors[label] ;
        }
    }
}
void merge(int arr[][3], int l, int m, int r) 
{ 
    int i, j, k; 
    int n1 = m - l + 1; 
    int n2 =  r - m; 
  
    /* create temp arrays */
    int L[n1][3], R[n2][3]; 
  
    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++)
    { 
        L[i][0] = arr[l + i][0];
        L[i][1] = arr[l + i][1];
        L[i][2] = arr[l + i][2];
    }

    for (j = 0; j < n2; j++) 
    {
        R[j][0] = arr[m + 1+ j][0];
        R[j][1] = arr[m + 1+ j][1];
        R[j][2] = arr[m + 1+ j][2]; 
    }
  
    /* Merge the temp arrays back into arr[l..r]*/
    i = 0; // Initial index of first subarray 
    j = 0; // Initial index of second subarray 
    k = l; // Initial index of merged subarray 
    while (i < n1 && j < n2) 
    { 
        if (L[i][1] <= R[j][1]) 
        { 
            arr[k][0] = L[i][0];
            arr[k][1] = L[i][1];
            arr[k][2] = L[i][2];
            i++; 
        } 
        else
        { 
            arr[k][0] = R[j][0];
            arr[k][1] = R[j][1];
            arr[k][2] = R[j][2]; 
            j++; 
        } 
        k++; 
    } 
  
    /* Copy the remaining elements of L[], if there 
       are any */
    while (i < n1) 
    { 
        arr[k][0] = L[i][0];
        arr[k][1] = L[i][1];
        arr[k][2] = L[i][2];
        i++; 
        k++; 
    } 
  
    /* Copy the remaining elements of R[], if there 
       are any */
    while (j < n2) 
    { 
        arr[k][0] = R[j][0];
        arr[k][1] = R[j][1];
        arr[k][2] = R[j][2]; 
        j++; 
        k++; 
    } 
} 
  
/* l is for left index and r is right index of the 
   sub-array of arr to be sorted */
void mergeSort(int arr[][3], int l, int r) 
{ 
    if (l < r) 
    { 
        // Same as (l+r)/2, but avoids overflow for large l and h 
        int m = l+(r-l)/2; 
        // Sort first and second halves 
        mergeSort(arr, l, m); 
        mergeSort(arr, m+1, r); 
        merge(arr, l, m, r); 
    } 
} 

void imageCb(const sensor_msgs::ImageConstPtr& msg)
{
    Mat image;
    cv_bridge::CvImagePtr cv_ptr;

    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        image = cv_ptr->image;
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    if( !image.data ) { printf("Error loading A \n"); return ; }
    // 1 4
    // 2 3
    imshow("Original", image);//Show Original Image

    Mat dst1(image.size(),CV_8UC1,Scalar(0));//dst1 is a black and white image of size of image
    Mat dst_grad(dst1.size(),CV_8UC1,Scalar(0));

    for(int i=0;i< image.rows;i++) 
    {
        for(int j=0;j<image.cols; j++)
        {
            if( j<min(y_1,y2) || j >max(y3,y4) || i<image.rows/3 )
            { 
                image.at<Vec3b>(i,j)[0]=0;
                image.at<Vec3b>(i,j)[1]=0;
                image.at<Vec3b>(i,j)[2]=0;
            }
        }
    }

    // Mat image1=image.clone();
    // warpPerspective(image1,image,h,image.size()); //Apply top view and store result into 'image'    
    // imshow("IPM",image); 
    cvtColor(image,dst1,CV_BGR2GRAY); //black BGR image to grayscale image

    //gradient_thresholding(image, dst1);
    imshow("2B-G",dst1);

    gradient_thresholding_horizontal(dst1,image, dst_grad);
    for( int i=dst_grad.rows-1  ;i>dst_grad.rows-20;i--)
        for( int j=0;j<dst_grad.cols;j++)
        {
              dst_grad.at<uchar>(i,j)=0;
        }

    imshow("Gradient_vertical",dst_grad); //show image after gradient thresholding
    //Return Structures required for the function
    Mat dst_labels(dst_grad.size(),CV_32S);
    Mat dst_stats(dst_grad.size(),CV_32S);
    Mat dst_centroid(dst_grad.size(),CV_64F);

    int nlabels=connectedComponentsWithStats(dst_grad,dst_labels, dst_stats, dst_centroid, 8, CV_32S );
    //Area based thresholding with defined ROI stored into dst_clustering
    Mat dst_clustering(dst_grad.size(),CV_8UC1,Scalar(0));
    for(int i=0;i<dst_clustering.rows;i++)
    {
        for(int j=0;j<dst_clustering.cols;j++)
        {
            //if the area of the cluster at (i,j) lies between 200,2000 then make it white, else remains black
            int label= dst_labels.at<int>(i,j);
            int centroid_y=dst_centroid.at<double>(label,0);
            if(dst_grad.at<uchar>(i,j)==0) continue; //if pixel is black it remians black
            if(dst_stats.at<int>(label, CC_STAT_AREA)>area_thresh ) dst_clustering.at<uchar>(i,j)=255;
        }
    }

    //Apply a second connected components with stats to get the blobs which have a certain area. Updated data goes into the same images as before
    nlabels=connectedComponentsWithStats(dst_clustering,dst_labels, dst_stats, dst_centroid, 8, CV_32S );
   
    //ONLY FOR DISPLAY
    Mat tp(dst_grad.size(),CV_8UC3);
    colour(tp,dst_labels,nlabels);
    imshow("After_labels_area_thresh",tp);
    //ONLY FOR DISPLAY
   
    //Create a vector structure where you can access each point in a label as -- cluster(LABEL,<POINT_NUMBER>) eg to access the 24 member of 3rd cluster write cluster(3,24)
    vector<vector<Point> > cluster(nlabels-1);
    for(int i=0;i<dst_clustering.rows;i++)
    {
        for(int j=0;j<dst_clustering.cols;j++)
        {
            if( dst_clustering.at<uchar>(i,j)==0 ) continue;
            int label= dst_labels.at<int>(i,j);
            cluster[label-1].push_back(Point(i,j));
        }
    }
    float *theta_array;
    theta_array= PCa(cluster); //get the value of all theta in the theta_array
    
    int j=0;
    for(int i=0;i<nlabels-1;i++)
    {
        if(theta_array[i]<=90+lateral_thresh && theta_array[i]>=90-lateral_thresh ) j++;
    }

    int cluster_x[j][3]; //0->label 1->x coordinate from origin 2->(value is different originally, becomes same if connected)

    int count=0;
    for(int i=0;i<nlabels-1;i++)
    {
        //Store only the lateral lane markings
        if(theta_array[i]<=90+lateral_thresh && theta_array[i]>=90-lateral_thresh )
        {
            cluster_x[count][0]=i+1;
            cluster_x[count][1]=dst_centroid.at<double>(i+1,1);
            cluster_x[count][2]=count;
            count++;
        }    
    }  
    //Now we have count number of objets in the array out_data
    //sort the x coordinates in ascenting order
    mergeSort(cluster_x,0,count-1);
    for(int i=0;i<j;i++)
    {
        cluster_x[i][2]=i;
    } 
    cout << endl << "--------For Debugging purpose--------" << endl ;
    //print cluster data
    for(int i=0;i<count;i++)
    {
        cout<< cluster_x[i][0] << " " << cluster_x[i][1] << " " << cluster_x[i][2] << endl;  
    }

    int x_cluster_number=count*2;
    for(int i=0;i<count;i++)
    {
        if(i==0)
        {
            if(fabs(cluster_x[i+1][1]-cluster_x[i][1]) <= lat_cluster_thresh) 
            {
                cluster_x[i+1][2]=cluster_x[i][2];
                x_cluster_number--;
            }
            continue;
        }

        if(i==count-1)
        {
            if(fabs(cluster_x[i-1][1]-cluster_x[i][1]) <= lat_cluster_thresh) 
            {
                cluster_x[i][2]=cluster_x[i-1][2];
                x_cluster_number--;
            }
            continue;
        }

        if(fabs(cluster_x[i-1][1]-cluster_x[i][1]) <= lat_cluster_thresh)
        {   
            cluster_x[i][2]=cluster_x[i-1][2];
            x_cluster_number--;
        }

        if(fabs(cluster_x[i+1][1]-cluster_x[i][1]) <= lat_cluster_thresh) 
        {
            cluster_x[i+1][2]=cluster_x[i][2];
            x_cluster_number--;
        }
    }
    
    x_cluster_number=x_cluster_number/2;
    
    //print cluster data
    for(int i=0;i<count;i++)
    {
        cout<< cluster_x[i][0] << " " << cluster_x[i][1] << " " << cluster_x[i][2] << endl;  
    }

    //Random Colours vector definition DELETE FROM FINAL CODE
    vector<Vec3b> colors(count);
    for(int i=0;i<count;i++)
    {
        colors[i]=Vec3b((rand()%255),(rand()%155),(rand()%255));
    }

    Mat show(dst_grad.size(),CV_8UC3,Scalar(0,0,0)); //Define an image show which will only show horizontal clusters as coloured
    //Assigning each blob a colour
    for(int i=0;i<show.rows;i++)
    {
        for(int j=0;j<show.cols;j++)
        {
            int label= dst_labels.at<int>(i,j);
            if(label==0) 
            {
                show.at<Vec3b>(i,j)=Vec3b(0,0,0);
                continue;
            }
            int l=-1;
            for(int i=0;i<count;i++)
            {
                if(cluster_x[i][0]==label) 
                {
                    l=i; 
                    break;
                }
            }
            if(l<0) 
            {
                show.at<Vec3b>(i,j)=Vec3b(255,255,255);
                continue;
            }
            if(count==1)
            {
                show.at<Vec3b>(i,j)=colors[0] ;
                continue;
            }
            show.at<Vec3b>(i,j)=colors[abs(cluster_x[l][2]+180)%(count-1)] ;
        }
    }

    if(x_cluster_number==0)
    {
        cout<< "No Stopline candidate" << endl;
        cout << "------------------------------------------------------------------------------" << endl;
        imshow("Detected horizontal clusters with STOPLINE",show);
        singlePos.x = -1 ;
        singlePos.y = -1 ;
        singlePos.theta = -1 ;
        waitKey(1);
        return;
    }

    //to store all the unique numbers of cluster_x[][2] create an array
    int unique[x_cluster_number]={-1};
    int t=0;
    for(int i=0;i<count;i++)
    {
        if(i==0) unique[t]=cluster_x[i][2];
        if(cluster_x[i][2]>unique[t])
        {
            t++;
            unique[t]=cluster_x[i][2];
        }
    }
    cout << "unique: ";
    for(int i=0;i<x_cluster_number;i++)
    {
        cout << unique[i] << " ";
    }
    cout << endl;
    
    //to store the potential candidates in the data strutcure out_data
    CD *out_data;
    out_data= (CD *)malloc(sizeof(CD)*x_cluster_number);
    
    for(int i=0;i<x_cluster_number;i++)
    {
        int centroid_x=0,centroid_y=0,t=0;
        float theta=0;
        //store the leftmost and rightmost values among all blobs of a cluster. length of cluster=(rightmost_centroid_y+width/2)-leftmost_y
        int leftmost_y=image.cols,rightmost_centroid_y=0;
        int label_rightmost_centroid_y;
        for(int j=0;j<count;j++)
        {
            if(cluster_x[j][2]==unique[i])
            {
                int label=cluster_x[j][0];
                centroid_x+=dst_centroid.at<double>(label,1);
                centroid_y+=dst_centroid.at<double>(label,0);
                theta+=theta_array[label-1];
                if(dst_stats.at<int>(label,CC_STAT_LEFT)<leftmost_y) leftmost_y=dst_stats.at<int>(label,CC_STAT_LEFT);
                if(dst_centroid.at<double>(label,0)>rightmost_centroid_y) 
                {
                    rightmost_centroid_y=dst_centroid.at<double>(label,0);
                    label_rightmost_centroid_y=label;
                }
                t++; //t is the number of blobs in the cluster, divide all by t to get average value of each component
            }
        }
        //Take average values of centroid and theta
        centroid_x=(int)(centroid_x/t);
        centroid_y=(int)(centroid_y/t);
        theta=theta/(float)(t);

        (*(out_data +i)).centroid_y= centroid_y;
        (*(out_data +i)).centroid_x= centroid_x;
        (*(out_data +i)).theta=theta;
        (*(out_data +i)).length= ( (rightmost_centroid_y  +  (dst_stats.at<int>(label_rightmost_centroid_y,CC_STAT_WIDTH)/2.0)) -leftmost_y);
    }  
    cout <<"--------For Debugging purpose--------" << endl ;
    cout << endl << "Super Cluster Data: " << endl ;
    for(int i=0;i<x_cluster_number;i++)
    {
       
        cout << i << endl;
        cout << "Centroid= " << (*(out_data +i)).centroid_x << " " << (*(out_data +i)).centroid_y<< endl;
        cout<<"Theta= "<< (*(out_data +i)).theta << endl;
        cout<<"Length= "<< (*(out_data +i)).length<< endl << endl;
    } 

    //Length Based Thresholding
    int lane_width;
    for(int i=0;i<x_cluster_number;i++)
    {
        lane_width=fabs(y_1-y3 +(*(out_data +i)).centroid_x*( 1.0*(y2-y_1)/(x2-x1) - 1.0*(y3-y4)/(x3-x4) ));
        if(  ((*(out_data +i)).length)>=(fabs(lane_width)/2.0) ) (*(out_data +i)).l_thresh=true;
        else  (*(out_data +i)).l_thresh=false;
    }
    //Store the theta of Horizon Line in degrees in the variable ctheta
    float ctheta=0;
    // if( atan2(cy2-cy1,cx2-cy1)<0 ) ctheta=atan2(cy2-cy1,cx2-cy1)+3.14159265;
    // else ctheta=atan2(cy2-cy1,cx2-cy1);
    // ctheta=ctheta*(180/3.1415);
    // cout<<"Ctheta"<<ctheta<<endl;

    int flag_no_stopline=1;
    for(int i=0;i<x_cluster_number;i++)
    {
        if( (*(out_data +i)).l_thresh==true ) //check g_thresh only if l_thresh is true
        {
            float g=fabs(  sin(  (ctheta-(*(out_data +i)).theta)*(3.1415/180.0)  )  );
            if(g>g_thresh) 
            {
                (*(out_data +i)).g=g;
                flag_no_stopline=0;
            }
            else (*(out_data +i)).g=-1;
        }
        else (*(out_data +i)).g=-1;

    }
    if(flag_no_stopline==0)
    {
        cout << endl << "Stopline Found" << endl;
        int stop_x=0;
        int stopline =0;
        for(int i=0;i<x_cluster_number;i++)
        {
            cout<<"G-Value :"<<(*(out_data +i)).g<<endl;
            if((*(out_data +i)).g>=0)
            {
                if((*(out_data +i)).centroid_x>stop_x)
                {
                    stop_x=(*(out_data +i)).centroid_x;
                    stopline=i;
                }
            }
        }
        cout<<"Stopline"<<stopline<<endl;
        //Plot The Centroid of cluster as green
        if(isvalid(show,((*(out_data +stopline)).centroid_x ),((*(out_data +stopline)).centroid_y)))
        {
            show.at<Vec3b>((int)((*(out_data +stopline)).centroid_x),(int)((*(out_data +stopline)).centroid_y ))[0]= 0;
            show.at<Vec3b>((int)((*(out_data +stopline)).centroid_x),(int)((*(out_data +stopline)).centroid_y ))[1]= 255;
            show.at<Vec3b>((int)((*(out_data +stopline)).centroid_x),(int)((*(out_data +stopline)).centroid_y ))[2]= 0;
        }
        //Horizontal line case
        if((*(out_data +stopline)).theta<=90+0.1 && (*(out_data +stopline)).theta>=90-0.1)
        {
            for(int j=((*(out_data +stopline)).centroid_y )- 200; j<  ((*(out_data +stopline)).centroid_y )+ 200;j++)
            {
                show.at<Vec3b>((*(out_data +stopline)).centroid_x ,j)=Vec3b(0,0,255);
            }
        }
        //Plot the line(eigenvector of the cluster)
        for(float x=((*(out_data +stopline)).centroid_x )- 20; x<  ((*(out_data +stopline)).centroid_x )+ 20;x=x+0.1)
        {
            float m= tan((*(out_data +stopline)).theta*(3.1415/180.0));
            float y= ((*(out_data +stopline)).centroid_y ) + m*(x-((*(out_data +stopline)).centroid_x));
            if(isvalid(show,x,y))
            {
                show.at<Vec3b>((int)x,(int)y)[0]= 0;
                show.at<Vec3b>((int)x,(int)y)[1]= 0;
                show.at<Vec3b>((int)x,(int)y)[2]= 255;
            }
        }
        imshow("Detected horizontal clusters with STOPLINE",show);
        singlePos.x = (*(out_data +stopline)).centroid_x ;
        singlePos.y = (*(out_data +stopline)).centroid_y ;        
        singlePos.theta =  (*(out_data +stopline)).theta ;
    }
    else
    {
        cout<<"NO STOPLINE"<<endl;
        imshow("Detected horizontal clusters with STOPLINE",show);
        singlePos.x = -1 ;
        singlePos.y = -1 ;
        singlePos.theta = -1 ;    
    }
    cout << "------------------------------------------------------------------------------" << endl;
    waitKey(1);
    return;
} 
void lane_callback(const geometry_msgs::Polygon::ConstPtr& lanes)
{
    geometry_msgs::Point32 temp;
    temp=lanes->points[0];
    x1=temp.x;
    y_1=temp.y;

    temp=lanes->points[1];
    x2=temp.x;
    y2=temp.y;

    temp=lanes->points[2];
    x3=temp.x;
    y3=temp.y;

    temp=lanes->points[3];
    x4=temp.x;
    y4=temp.y;
}


int main(int argc, char** argv)
{
    cout<<"Running"<<endl;
    ros::init(argc, argv, "Stopline");
    ros::NodeHandle nh;

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber image_sub;    
    image_sub = it.subscribe("Image", 1, &imageCb);
    
    ros::Subscriber lanesub = nh.subscribe("lane_data",1000,lane_callback);
    ros::Publisher pub = nh.advertise<geometry_msgs::Pose2D>("Topic", 1000);


    ros::Rate r(1);
    while(ros::ok())
    {
        pub.publish(singlePos);
        cout<<"Published"<<endl;
        r.sleep();
        ros::spinOnce();
    }
    return 0;
}


    