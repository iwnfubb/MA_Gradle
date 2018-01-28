package algorithms;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import utils.Utils;

import java.util.ArrayList;
import java.util.List;

public class DiffMotionDetector {
    Mat background_gray;
    private int threshold = 10;
    List<Rect> history = new ArrayList<>();
    private boolean isBackgroundSet = false;
    private Rect last_motion;
    boolean isObjectMoving = false;
    private Mat thresholdMat = new Mat();
    private double backgroundDensity = 0;
    private boolean trigger = false;
    private int counter;



    public DiffMotionDetector() {
        background_gray = new Mat();
    }

    public void setBackground(Mat frame) {
        Imgproc.cvtColor(frame, background_gray, Imgproc.COLOR_BGR2GRAY);
    }

    public Mat getBackground() {
        return background_gray;
    }

    private Mat returnMask(Mat frame) {
        backgroundDensity = 0;
        if (frame.empty()) {
            return new Mat();
        }
        Mat image_gray = new Mat();
        Imgproc.cvtColor(frame, image_gray, Imgproc.COLOR_BGR2GRAY);
        Mat delta_image = new Mat();
        Core.absdiff(background_gray, image_gray, delta_image);
        Mat threshold_image = new Mat();

        Imgproc.threshold(delta_image, threshold_image, threshold, 255, Imgproc.THRESH_BINARY);

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));

        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();
        int connectivity = 4;

        Mat kernelErode = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.erode(threshold_image, threshold_image, kernelErode);
        Mat kernelDalate = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.dilate(threshold_image, threshold_image, kernelDalate);
        //Imgproc.morphologyEx(threshold_image, threshold_image, Imgproc.MORPH_CLOSE, kernel);
        Imgproc.connectedComponentsWithStats(threshold_image, labels, stats, centroids,
                connectivity, CvType.CV_32S);
        double sum = 0;
        for (int i = 0; i < stats.rows(); i++) {
            sum += stats.get(i, 4)[0];
        }

        if (BinaryMaskAnalyser.returnNumberOfContours(threshold_image) > 0) {
            Rect currentMotion = BinaryMaskAnalyser.returnMaxAreaRectangle(threshold_image);
            if (currentMotion != null) {
                history.add(currentMotion);
                isObjectMoving = isObjectMoving(currentMotion);
                last_motion = currentMotion;
                backgroundDensity = stats.get(0, 4)[0] / sum;
                updateBackgroundImage(currentMotion, image_gray);
                //Rect maxRect = getMaxRect(history);
                //updateBackgroundImage2(maxRect, image_gray);
            }
        }

        System.out.println("##### BackgroundDensity: " + backgroundDensity);
        threshold_image.copyTo(thresholdMat);
        return threshold_image;
    }


    public void updateBackgroundImage(Rect currentMotion, Mat image_gray) {
        if (backgroundDensity < 0.8) {
            history.removeAll(history);
            image_gray.copyTo(background_gray);
            trigger = true;
            counter = 0;
        }

        if (trigger) {
            counter++;
            if (counter >= 30) {
                history.removeAll(history);
                image_gray.copyTo(background_gray);
                trigger = false;
                counter = 0;
            }
        }

        for (int i = 0; i < history.size(); i++) {
            Rect r = history.get(i);
            if (!Utils.overlaps(r, currentMotion)) {
                Mat imageROI = new Mat(image_gray, r);
                imageROI.copyTo(background_gray.colRange(r.x, r.x + imageROI.cols()).rowRange(r.y, r.y + imageROI.rows()));
                history.remove(i);
            }
        }

    }

    public void updateBackgroundImage2(Rect currentMotion, Mat image_gray) {
        //Mat imageROI = new Mat(image_gray, r);
        Rect topRect = new Rect(0, 0, image_gray.width(), currentMotion.y);
        Mat topImage = new Mat(image_gray, topRect);

        Rect bottomRect = new Rect(0, currentMotion.y + currentMotion.height,
                image_gray.width(), image_gray.height() - (currentMotion.y + currentMotion.height));
        Mat bottomImage = new Mat(image_gray, bottomRect);

        Rect leftRect = new Rect(0, 0, currentMotion.x, image_gray.height());
        Mat leftImage = new Mat(image_gray, leftRect);

        Rect rightRect = new Rect(currentMotion.x + currentMotion.width, 0,
                image_gray.width() - (currentMotion.x + currentMotion.width), image_gray.height());
        Mat rightImage = new Mat(image_gray, rightRect);


        topImage.copyTo(background_gray.rowRange(0, topImage.rows()));
        System.out.println("top");
        bottomImage.copyTo(background_gray.rowRange(currentMotion.y + currentMotion.height, background_gray.rows()));
        System.out.println("bottom");
        leftImage.copyTo(background_gray.colRange(0, leftImage.cols()));
        System.out.println("left");
        rightImage.copyTo(background_gray.colRange(currentMotion.x + currentMotion.width, background_gray.cols()));
        System.out.println("right");
    }

    private Rect getMaxRect(List<Rect> list) {
        if (list.size() > 1) {
            int minX = 1280;
            int minY = 720;
            int maxX = 0;
            int maxY = 0;
            for (Rect r : list) {
                if (r.x < minX) {
                    minX = r.x;
                }
                if (r.x + r.width > maxX) {
                    maxX = r.x + r.width;
                }
                if (r.y < minY) {
                    minY = r.y;
                }
                if (r.y + r.height > maxY) {
                    maxY = r.y + r.height;
                }
            }
            return new Rect(minX, minY, maxX - minX, maxY - minY);
        } else {
            return new Rect(0, 0, background_gray.width(), background_gray.height());
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
                            new Scalar(255, 0, 0), 2);
                }
            }
        }
        return frame;
    }

    public boolean isObjectMoving(Rect current_motion) {
        if (last_motion == null) {
            return false;
        }
        return MovingDetector.isObjectMoving(current_motion, last_motion);
    }
}
