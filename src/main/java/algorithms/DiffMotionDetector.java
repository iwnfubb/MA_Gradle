package algorithms;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractorKNN;
import org.opencv.video.Video;
import utils.Parameters;
import utils.Utils;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class DiffMotionDetector {
    Mat background_gray;
    Mat diff_gray = new Mat();
    Mat thershold_gray = new Mat();
    Mat erodela_gray = new Mat();
    Mat update_bg = new Mat();
    Mat notupdate_bg = new Mat();

    private int threshold = 10;
    List<Rect> history = new ArrayList<>();
    Rect history_knn = new Rect(0, 0, 0, 0);
    Mat knn_mask = new Mat();
    public boolean isBackgroundSet = false;
    Mat thresholdMat = new Mat();
    public double backgroundDensity = 0;
    private boolean trigger = false;
    private boolean trigger_KNN = false;
    private int counter = 0;
    private int counter_KNN = 0;

    BackgroundSubtractorKNN backgroundSubtractorKNN;
    Person.Persons personsList;


    public DiffMotionDetector() {
        background_gray = new Mat();
        personsList = new Person.Persons(Parameters.movementMaximum, Parameters.movementMinimum, Parameters.badLimit);
        backgroundSubtractorKNN = Video.createBackgroundSubtractorKNN();
        backgroundSubtractorKNN.setHistory(100);
        backgroundSubtractorKNN.setDetectShadows(true);
    }


    public void setBackground(Mat frame) {
        Imgproc.cvtColor(frame, background_gray, Imgproc.COLOR_BGR2GRAY);
    }

    private Mat returnMask_MOG(Mat frame) {
        if (frame.empty()) {
            return new Mat();
        }

        Mat kernelErode3 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Mat kernelDalate5 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();
        int connectivity = 8;

        Mat mog2Mask = new Mat();
        backgroundSubtractorKNN.applyOutsideRect(frame, mog2Mask, 0.01, history_knn);
        if (backgroundDensity < 0.8) {
            trigger_KNN = true;
            counter_KNN = 0;
        }

        if (trigger_KNN) {
            counter_KNN++;
            backgroundSubtractorKNN.apply(frame, mog2Mask, 0.01);
            if (counter_KNN >= 30) {
                history.removeAll(history);
                personsList.persons.removeAll(personsList.persons);
                trigger_KNN = false;
                counter_KNN = 0;
            }
        }


        Imgproc.threshold(mog2Mask, mog2Mask, 128, 255, Imgproc.THRESH_BINARY);
        Imgproc.erode(mog2Mask, mog2Mask, kernelErode3);
        Imgproc.dilate(mog2Mask, mog2Mask, kernelDalate5);
        Imgproc.connectedComponentsWithStats(mog2Mask, labels, stats, centroids, connectivity, CvType.CV_32S);

        backgroundDensity = 0;
        double sum = 0;
        for (int i = 0; i < stats.rows(); i++) {
            sum += stats.get(i, 4)[0];
        }

        if (BinaryMaskAnalyser.returnNumberOfContours(mog2Mask) > 0) {
            Rect currentMotion = BinaryMaskAnalyser.returnMaxAreaRectangle(mog2Mask);
            if (currentMotion != null) {
                history.removeAll(history);
                history.add(currentMotion);
                history_knn = currentMotion;
            } else {
                history_knn = new Rect(0, 0, 0, 0);
            }
        }
        backgroundDensity = stats.get(0, 4)[0] / sum;

        System.out.println("##### BackgroundDensity: " + backgroundDensity);
        mog2Mask.copyTo(thresholdMat);
        backgroundSubtractorKNN.getBackgroundImage(background_gray);
        return mog2Mask;
    }

    private Mat returnMask(Mat frame) {
        frame.copyTo(update_bg);
        frame.copyTo(notupdate_bg);
        Mat kernelErode3 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Mat kernelDalate3 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Mat kernelDalate5 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Mat kernelDalate7 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(7, 7));
        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();
        int connectivity = 8;
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));


        Mat mog2Mask = new Mat();
        backgroundSubtractorKNN.apply(frame, mog2Mask, 0.01);
        Mat shadow_binary_image = new Mat();
        Imgproc.threshold(mog2Mask, shadow_binary_image, 128, 255, Imgproc.THRESH_TOZERO_INV);

        Imgproc.threshold(mog2Mask, mog2Mask, 128, 255, Imgproc.THRESH_BINARY);
        Imgproc.erode(mog2Mask, mog2Mask, kernelErode3);
        Imgproc.dilate(mog2Mask, mog2Mask, kernelDalate5);
        Imgproc.connectedComponentsWithStats(mog2Mask, labels, stats, centroids, connectivity, CvType.CV_32S);
        mog2Mask.copyTo(knn_mask);

        Imgproc.threshold(shadow_binary_image, shadow_binary_image, 1, 255, Imgproc.THRESH_BINARY);

        backgroundDensity = 0;
        if (frame.empty()) {
            return new Mat();
        }
        Mat image_gray = new Mat();
        Imgproc.cvtColor(frame, image_gray, Imgproc.COLOR_BGR2GRAY);

        Mat delta_image = new Mat();
        Core.absdiff(background_gray, image_gray, delta_image);
        delta_image.copyTo(diff_gray);
        Mat threshold_image = new Mat();

        Imgproc.threshold(delta_image, threshold_image, threshold, 255, Imgproc.THRESH_BINARY);
        threshold_image.copyTo(thershold_gray);

        Imgproc.erode(shadow_binary_image, shadow_binary_image, kernelErode3);
        Imgproc.dilate(shadow_binary_image, shadow_binary_image, kernelDalate3);
        Imgproc.connectedComponentsWithStats(shadow_binary_image, labels, stats, centroids, connectivity, CvType.CV_32S);

        Core.subtract(threshold_image, shadow_binary_image, threshold_image);

        Imgproc.erode(threshold_image, threshold_image, kernelErode3);
        Imgproc.dilate(threshold_image, threshold_image, kernelDalate5);
        threshold_image.copyTo(erodela_gray);
        //Imgproc.morphologyEx(threshold_image, threshold_image, Imgproc.MORPH_CLOSE, kernel);
        Imgproc.connectedComponentsWithStats(threshold_image, labels, stats, centroids, connectivity, CvType.CV_32S);

        double sum = 0;
        for (int i = 0; i < stats.rows(); i++) {
            sum += stats.get(i, 4)[0];
        }

        if (BinaryMaskAnalyser.returnNumberOfContours(threshold_image) > 0) {
            Rect currentMotion = BinaryMaskAnalyser.returnMaxAreaRectangle(threshold_image);
            //ArrayList<Rect> rects = BinaryMaskAnalyser.notMaxAreaRectangle(threshold_image);
            //history.addAll(rects);
            if (currentMotion != null) {
                history.add(currentMotion);
                backgroundDensity = stats.get(0, 4)[0] / sum;
                updateBackgroundImage(currentMotion, image_gray);
            }
        }
        if (BinaryMaskAnalyser.returnNumberOfContours(mog2Mask) > 0) {
            Rect currentMotion = BinaryMaskAnalyser.returnMaxAreaRectangle(mog2Mask);
            if (currentMotion != null) {
                history_knn = currentMotion;
            } else {
                history_knn = new Rect(0, 0, 0, 0);
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
                image_gray.copyTo(background_gray);
                //updateBackgroundImage2(currentMotion, image_gray);
                history.removeAll(history);
                personsList.persons.removeAll(personsList.persons);
                trigger = false;
                counter = 0;
            }
        }

        Iterator<Rect> iterator = history.iterator();
        while (iterator.hasNext()) {
            Rect r = iterator.next();
            if (!Utils.overlaps(r, currentMotion)) {
                Mat imageROI = new Mat(image_gray, r);
                imageROI.copyTo(background_gray.colRange(r.x, r.x + imageROI.cols()).rowRange(r.y, r.y + imageROI.rows()));
                drawImageWithRect(update_bg, r, Parameters.color_red);
                iterator.remove();
            }else {
                drawImageWithRect(notupdate_bg, r, Parameters.color_white);
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

    public Mat getDiffDetector_MOG(Mat currentFrame) {
        Mat frame = new Mat();
        if (!currentFrame.empty()) {
            currentFrame.copyTo(frame);
            if (!isBackgroundSet) {
                setBackground(frame);
                isBackgroundSet = true;
            }
            returnMask_MOG(frame);
        }
        return frame;
    }

    private void drawRect(Mat img, MatOfRect matOfRect, Scalar color) {
        List<Rect> rects = matOfRect.toList();
        for (Rect r : rects) {
            Imgproc.rectangle(img, r.tl(), r.br(), color, 5);
        }
    }


    private Mat drawImageWithRect(Mat input, Rect bestRect, Scalar color) {
        MatOfRect matOfRect = new MatOfRect();
        matOfRect.fromArray(bestRect);
        drawRect(input, matOfRect, color);
        return input;
    }
}
