#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <stdio.h>
#include <time.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame , Mat screen, int mode);
void overlayImage(const Mat &background, const Mat &foreground, Mat &output, Point2i location);

/** Global variables */
String face_cascade_name = "haarcascade_frontalface_alt.xml";

CascadeClassifier face_cascade;

int main(int argc, char *argv[])
{
    VideoCapture cap(0);
    if(!cap.isOpened()){
        return -1;
    }

    Mat screen;
    namedWindow("screen",1);

    int iSliderValue1 = 3;
    createTrackbar("SmileMeter", "screen", &iSliderValue1, 5);

    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };

    srand(time(NULL));
    int mode = 0;

    while(true){
        Mat frame;
        cap >> frame;
        cvtColor(frame, screen, CV_BGR2HSV);

        MatIterator_<Vec3b> it = screen.begin<Vec3b>(), it_end = screen.end<Vec3b>();
        for(; it != it_end; ++it){
            Vec3b& pixel = *it;
            pixel[1] = pixel[1] * iSliderValue1 / 5;
        }

        cvtColor(screen, screen, CV_HSV2BGR);

        if(iSliderValue1 < 3){
            if(mode == 0){
                mode = rand()%4 + 1; // 1 = eyebrow, 2 = scream face, 3 = nose, 4 = wig
            }
            detectAndDisplay( frame , screen, mode);
        } else {
            mode = 0;
            imshow("screen", screen);
        }
        int c = waitKey(100);
        if( (char)c == 27 ) { break; } // escape
    }

    return 0;
}

void detectAndDisplay( Mat frame , Mat screen, int mode){
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );

    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(80, 80) );

    for ( size_t i = 0; i < faces.size(); i++ ) {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        //ellipse( screen, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 0 ), 4, 8, 0 );

        Mat faceROI = frame_gray( faces[i] );

        switch (mode){
        case 1:{
                Mat eyebrow = imread("Eyebrow.png", IMREAD_UNCHANGED);
                Mat re_eyebrow;
                resize( eyebrow, re_eyebrow, Size(faces[i].width*0.8, faces[i].width * eyebrow.rows / eyebrow.cols) );

                Mat imgROI = screen(faces[i]);
                overlayImage(imgROI, re_eyebrow, imgROI, Point(faces[i].width*0.1, faces[i].width * 1/8));
            }
            break;
        case 2:{
                Mat scream = imread("ScreamFace.png", IMREAD_UNCHANGED);
                Mat re_scream;
                resize( scream, re_scream, Size(faces[i].width, faces[i].height*1.35) );

                overlayImage(screen, re_scream, screen, Point(faces[i].x, faces[i].y - faces[i].height*0.18));
            }
            break;
        case 3:{
				Point noseP((faces[i].width / 3), (faces[i].height *0.4));
				Mat rednose = imread("rednose.png", IMREAD_UNCHANGED);
				Mat rednose_resized, rednose_mask_resized;
				resize(rednose, rednose_resized, cv::Size(faces[i].width *0.36 , faces[i].width *0.36));
				Mat imgROI = screen(faces[i]);
				overlayImage(imgROI, rednose_resized, imgROI, noseP);
            }
            break;
        case 4:{
                Mat wig = imread("wig.png", IMREAD_UNCHANGED);
                Mat re_wig;
                resize( wig, re_wig, Size(faces[i].width*2, faces[i].height*2) );

                overlayImage(screen, re_wig, screen, Point(faces[i].x + faces[i].width/2 - re_wig.cols/2,
                                                           faces[i].y - re_wig.rows * 0.45));
            }
            break;
        default:
            break;
        }
    }

    putText(screen, "Smile more!", Point(10,30), FONT_HERSHEY_SIMPLEX , 1, Scalar(0,0,0));
    imshow( "screen", screen );
}

void overlayImage(const Mat &background, const Mat &foreground, Mat &output, Point2i location){
    background.copyTo(output);

    for(int y = max(location.y , 0); y < background.rows; ++y) {
        int fY = y - location.y;

        if(fY >= foreground.rows)
            break;

        for(int x = max(location.x, 0); x < background.cols; ++x) {
            int fX = x - location.x;

            if(fX >= foreground.cols)
                break;

            double opacity = ((double)foreground.data[fY * foreground.step + fX * foreground.channels() + 3]) / 255.;

            for(int c = 0; opacity > 0 && c < output.channels(); ++c) {
                unsigned char foregroundPx = foreground.data[fY * foreground.step + fX * foreground.channels() + c];
                unsigned char backgroundPx = background.data[y * background.step + x * background.channels() + c];
                output.data[y*output.step + output.channels()*x + c] = backgroundPx * (1.-opacity) + foregroundPx * opacity;
            }
        }
    }
}
