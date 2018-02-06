package algorithms;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import utils.Parameters;
import utils.Utils;

import java.util.ArrayList;
import java.util.List;

public class DiffMotionDetector {
    Mat background_gray;
    private int threshold = 10;
    List<Rect> history = new ArrayList<>();
    private boolean isBackgroundSet = false;
    Mat thresholdMat = new Mat();
    private double backgroundDensity = 0;
    private boolean trigger = false;
    private int counter;

    Person.Persons personsList;


    public DiffMotionDetector() {
        background_gray = new Mat();
        personsList = new Person.Persons(Parameters.movementMaximum, Parameters.movementMinimum, Parameters.movementTime);
    }


    public void setBackground(Mat frame) {
        if (Utils.activeShadowRemover) {
            Utils.calculateInvariant(frame).copyTo(frame);
        }
        Imgproc.cvtColor(frame, background_gray, Imgproc.COLOR_BGR2GRAY);
    }


    private Mat returnMask(Mat frame) {
        long startTime = System.currentTimeMillis();
        backgroundDensity = 0;
        if (frame.empty()) {
            return new Mat();
        }
        Mat image_gray = new Mat();
        if (Utils.activeShadowRemover) {
            Utils.calculateInvariant(frame).copyTo(frame);
        }
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
                backgroundDensity = stats.get(0, 4)[0] / sum;
                updateBackgroundImage(currentMotion, image_gray);
            }
        }
        System.out.println("##### BackgroundDensity: " + backgroundDensity);
        System.out.println("##### Time: " + (System.currentTimeMillis() - startTime));
        threshold_image.copyTo(thresholdMat);
        return threshold_image;
    }


    public void updateBackgroundImage(Rect currentMotion, Mat image_gray) {
        if (backgroundDensity < 0.9) {
            history.removeAll(history);
            image_gray.copyTo(background_gray);
            trigger = true;
            counter = 0;
        }

        if (trigger) {
            counter++;
            if (counter >= 30) {
                image_gray.copyTo(background_gray);
                //updateBackgroundImage2(currentMotion, image_gray);
                history.removeAll(history);
                personsList.persons.removeAll(personsList.persons);
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


    public Mat getDiffDetector(Mat currentFrame) {
        Mat frame = new Mat();
        if (!currentFrame.empty()) {
            currentFrame.copyTo(frame);
            if (!isBackgroundSet) {
                setBackground(frame);
                isBackgroundSet = true;
            }
            returnMask(frame);
        }
        return frame;
    }
}
