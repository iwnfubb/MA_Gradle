package algorithms;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import utils.Utils;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;

public class DiffMotionDetector {
    Mat background_gray;
    int threshold = 25;
    List<Rect> history = new ArrayList<>();


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
        Mat foreground_gray = new Mat();
        Imgproc.cvtColor(foregroundImage, foreground_gray, Imgproc.COLOR_BGR2GRAY);
        Mat delta_image = new Mat();
        Core.absdiff(background_gray, foreground_gray, delta_image);
        Mat threshold_image = new Mat();

        Imgproc.threshold(delta_image, threshold_image, threshold, 255, Imgproc.THRESH_BINARY);
        if (BinaryMaskAnalyser.returnNumberOfContours(threshold_image) > 0) {
            Rect currentMotion = BinaryMaskAnalyser.returnMaxAreaRectangle(threshold_image);
            if (currentMotion != null) {
                history.add(currentMotion);
                updateBackgroundImage(currentMotion, foreground_gray);
            }
        }
        return threshold_image;
    }

    public void updateBackgroundImage(Rect currentMotion, Mat foreground_gray) {
        for (int i = 0 ; i < history.size(); i++) {
            Rect r = history.get(i);
            if (!Utils.overlaps(r, currentMotion)) {
                Mat imageROI = new Mat(foreground_gray, r);
                imageROI.copyTo(background_gray.colRange(r.x, r.x + imageROI.cols()).rowRange(r.y, r.y + imageROI.rows()));
                history.remove(i);
            }
        }
    }
}
