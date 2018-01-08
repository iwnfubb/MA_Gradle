package algorithms;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class DiffMotionDetector {
    Mat background_gray;
    int threshold = 25;

    public DiffMotionDetector() {
        background_gray = new Mat();
    }

    public void setBackground(Mat frame) {
        Imgproc.cvtColor(frame, background_gray, Imgproc.COLOR_BGR2GRAY);
    }

    public Mat getBackground() {
        return background_gray;
    }

    public Mat returnMask(Mat foregroundImage) {
        if (foregroundImage.empty()) {
            return new Mat();
        }
        ;
        Mat foreground_gray = new Mat();
        Imgproc.cvtColor(foregroundImage, foreground_gray, Imgproc.COLOR_BGR2GRAY);
        Mat delta_image = new Mat();
        Core.absdiff(background_gray, foreground_gray, delta_image);
        Mat threshold_image = new Mat();
        Imgproc.threshold(delta_image, threshold_image, threshold, 255, Imgproc.THRESH_BINARY);
        return threshold_image;
    }
}
