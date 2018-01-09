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
    boolean isBackgroundSet = false;
    Rect last_motion;
    boolean isObjectMoving = false;
    Mat thresholdMat = new Mat();


    public DiffMotionDetector() {
        background_gray = new Mat();
    }

    public void setBackground(Mat frame) {
        Utils.convertImageByInvariantFeatures(frame).copyTo(frame);
        Imgproc.cvtColor(frame, background_gray, Imgproc.COLOR_BGR2GRAY);
    }

    public Mat getBackground() {
        return background_gray;
    }

    private Mat returnMask(Mat foregroundImage) {
        if (foregroundImage.empty()) {
            return new Mat();
        }
        Utils.convertImageByInvariantFeatures(foregroundImage).copyTo(foregroundImage);
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
                isObjectMoving = isObjectMoving(currentMotion);
                last_motion = currentMotion;
                updateBackgroundImage(currentMotion, foreground_gray);

            }
        }

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(20, 15));
        Imgproc.morphologyEx(threshold_image, threshold_image, Imgproc.MORPH_CLOSE, kernel);

        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();
        int connectivity = 8;
        Imgproc.connectedComponentsWithStats(threshold_image, labels, stats, centroids,
                connectivity, CvType.CV_32S);
        Mat kernelErode = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.erode(threshold_image, threshold_image, kernelErode);
        Mat kernelDalate = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(10, 10));
        Imgproc.dilate(threshold_image, threshold_image, kernelDalate);



        threshold_image.copyTo(thresholdMat);
        return threshold_image;
    }

    public void updateBackgroundImage(Rect currentMotion, Mat foreground_gray) {
        for (int i = 0; i < history.size(); i++) {
            Rect r = history.get(i);
            if (!Utils.overlaps(r, currentMotion)) {
                Mat imageROI = new Mat(foreground_gray, r);
                imageROI.copyTo(background_gray.colRange(r.x, r.x + imageROI.cols()).rowRange(r.y, r.y + imageROI.rows()));
                history.remove(i);
            }
        }
    }

    public Mat getDiffDetector(Mat currentFrame) {
        Mat frame = new Mat();
        if (!currentFrame.empty()) {
            currentFrame.copyTo(frame);
            if (!isBackgroundSet) {
                setBackground(frame);
                isBackgroundSet = true;
            }
            Mat frame_mask = returnMask(frame);
            Rect rect;
            if (BinaryMaskAnalyser.returnNumberOfContours(frame_mask) > 0) {
                rect = BinaryMaskAnalyser.returnMaxAreaRectangle(frame_mask);
                if (rect != null) {
                    Imgproc.rectangle(frame, new Point(rect.x, rect.y),
                            new Point(rect.x + rect.width, rect.y + rect.height),
                            new Scalar(0, 255, 0), 2);
                }
            }
        }
        return frame;
    }

    public boolean isObjectMoving(Rect current_motion) {
        if (last_motion == null)
            return false;
        return MovingDetector.isObjectMoving(current_motion, last_motion);
    }
}
