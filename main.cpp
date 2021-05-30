#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;
int distanceint ,i;

struct homme
{
    int dist;
    int x;
    int y;
};

homme hommes[100];
int nbhomme = 0;

void ResizeBoxes(Rect& box) {
    box.x += cvRound(box.width * 0.1);
    box.width = cvRound(box.width * 0.8);
    box.y += cvRound(box.height * 0.06);
    box.height = cvRound(box.height * 0.8);
}

int main(int argc, char** argv)
{
    vector<Point3d> img3d;
    Mat cameraMatrix;
    Mat frame;

    cameraMatrix.push_back(Point3d(6.5746697944293521e+002, 0, 3.8040000000000000e+002));
    cameraMatrix.push_back(Point3d(0, 6.5746697944293521e+002, 2.8850000000000000e+002));
    cameraMatrix.push_back(Point3d(0, 0, 1));

    img3d.push_back(Point3d(0, 0, 0));
    img3d.push_back(Point3d(60, 0, 0));
    img3d.push_back(Point3d(60, 175, 0));
    img3d.push_back(Point3d(0, 175, 0));

    vector<Point2d> image_points;
    Mat tvec;
    Mat rvec;
    Mat rotation_front;
    Mat world_position_front_cam;
    Mat dist_coeffs = Mat::zeros(4, 1, DataType<double>::type);
    vector<Rect> detections;
    Ptr<MultiTracker> multiTracker = MultiTracker::create();
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    VideoCapture cap;
    string file = "C:\\Users\\aterr\\source\\repos\\ProjectIA\\vtest.avi";
    cap.open(file);
    if (!cap.isOpened())
    {
        cout << "Can not open video stream: '" << (file.empty() ? "<camera>" : file) << "'" << endl;
        return 2;
    }
    int j = 0;
    MultiTracker trackers;
    for (;;)
    {
        j++;
        cap >> frame;
        if (frame.empty())
        {
            cout << "Finished reading: empty frame" << endl;
            break;
        }
        int64 t = getTickCount();
        if (j % 6 == 0 or j == 1) //detections
        {
            detections.clear();
            hog.detectMultiScale(frame, detections, 0, Size(8, 8), Size(16, 16), 1, 1);
            Ptr<MultiTracker> multiTrackerTemp = MultiTracker::create();
            multiTracker = multiTrackerTemp;
            nbhomme = 0;
            for (auto& detection : detections)
            {
                ResizeBoxes(detection);
                image_points.clear();
                image_points.push_back(Point2d(detection.x, detection.y));
                image_points.push_back(Point2d(detection.x+detection.width, detection.y));
                image_points.push_back(Point2d(detection.x+detection.width, detection.y + detection.height));
                image_points.push_back(Point2d(detection.x, detection.y + detection.height));
                multiTracker->add(TrackerCSRT::create(), frame, detection);
                solvePnP(img3d, image_points, cameraMatrix, dist_coeffs, rvec, tvec);
                vector<Point2d> imagePoints;
                Rodrigues(rvec, rotation_front);
                Mat rotation_inverse;
                transpose(rotation_front, rotation_inverse);
                world_position_front_cam = -rotation_inverse * tvec;
                distanceint = norm(Mat(world_position_front_cam));
                hommes[nbhomme].x = detection.x;
                hommes[nbhomme].y = detection.y;
                hommes[nbhomme++].dist = distanceint;
            }
        }
        
        else //Tracking
        {
            multiTracker->update(frame);
        }
        t = getTickCount() - t;
        {
            ostringstream buf;
            buf << "FPS: " << fixed << setprecision(1) << (getTickFrequency() / (double)t);
            putText(frame, buf.str(), Point(10, 30), FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 255), 2, LINE_AA);
        }

        for (i = 0; i < nbhomme; i++) {
            putText(frame, format("%.2f m", (float)hommes[i].dist /100) , 
            Point(hommes[i].x, hommes[i].y - 15), FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 255), 2, LINE_AA);
        }
        for (const auto& object : multiTracker->getObjects()) {
            rectangle(frame, object, Scalar(255, 0, 0), 2, 8);
            
        }
        imshow("People detector", frame);
        const char key = (char)waitKey(1);
    }
    return 0;
}
