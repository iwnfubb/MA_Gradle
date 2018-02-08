package imageprocess;

import algorithms.DiffMotionDetector;
import algorithms.KernelDensityEstimator;
import algorithms.Vibe;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractorKNN;
import org.opencv.video.BackgroundSubtractorMOG2;
import org.opencv.video.Video;
import org.opencv.videoio.VideoCapture;
import utils.Utils;

public class ImageProcess_MotionDetection {
    private VideoCapture capture;
    private Mat currentFrame = new Mat();
    private BackgroundSubtractorMOG2 pMOG2;
    private BackgroundSubtractorKNN pknn;
    private Size gaussianFilterSize = (new Size(3, 3));
    KernelDensityEstimator kde;
    Vibe vibe;
    DiffMotionDetector diffMotionDetector;

    public ImageProcess_MotionDetection(VideoCapture capture) {
        this.capture = capture;
        pMOG2 = Video.createBackgroundSubtractorMOG2();
        pMOG2.setHistory(100);
        pMOG2.setDetectShadows(true);
        pknn = Video.createBackgroundSubtractorKNN();
        pknn.setHistory(100);
        kde = new KernelDensityEstimator();
        kde.setN(10);
        vibe = new Vibe();
        diffMotionDetector = new DiffMotionDetector();
    }

    public Mat getOriginalFrame() {
        if (this.capture.isOpened()) {
            try {
                this.capture.read(currentFrame);
            } catch (Exception e) {
                System.err.println("Exception during the image elaboration: " + e);
            }
        }
        return currentFrame;
    }

    public Mat getGaussianMixtureModel() {
        Mat frame = new Mat();
        Mat blurFrame = new Mat();
        Mat shadow_binary_image =  new Mat();
        if (!currentFrame.empty()) {
            Imgproc.GaussianBlur(currentFrame, blurFrame, gaussianFilterSize, 0);
            pMOG2.apply(blurFrame, frame, 0.001);
            Imgproc.threshold(frame, shadow_binary_image, 128, 255, Imgproc.THRESH_TOZERO_INV);
            Imgproc.threshold(shadow_binary_image, shadow_binary_image, 1, 255, Imgproc.THRESH_BINARY);

            Mat kernelErode = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
            Imgproc.erode(shadow_binary_image, shadow_binary_image, kernelErode);
            Mat kernelDalate = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
            Imgproc.dilate(shadow_binary_image, shadow_binary_image, kernelDalate);
            //Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
            //Imgproc.morphologyEx(shadow_binary_image, shadow_binary_image, Imgproc.MORPH_CLOSE, kernel);
            Mat labels = new Mat();
            Mat stats = new Mat();
            Mat centroids = new Mat();
            int connectivity = 8;
            Imgproc.connectedComponentsWithStats(shadow_binary_image, labels, stats, centroids, connectivity, CvType.CV_32S);
        }
        return shadow_binary_image;
    }

    public Mat getKNNModel() {
        Mat frame = new Mat();
        Mat blurFrame = new Mat();
        if (!currentFrame.empty()) {
            Imgproc.GaussianBlur(currentFrame, blurFrame, gaussianFilterSize, 0);
            pknn.apply(blurFrame, frame, 0.1);
        }
        return frame;
    }

    public Mat getKDEModel() {
        Mat frame = new Mat();
        Mat blurFrame = new Mat();
        if (!currentFrame.empty()) {
            Imgproc.resize(currentFrame, blurFrame, new Size(currentFrame.width() / 2, currentFrame.height() / 2));
            Imgproc.GaussianBlur(blurFrame, blurFrame, gaussianFilterSize, 0);

            try {
                frame = kde.foregroundMask(blurFrame);
            } catch (KernelDensityEstimator.KernelDensityEstimatorException e) {
                log("Error");
                e.printStackTrace();
            }
        }
        return frame;
    }

    public Mat getVibeModel() {
        Mat frame = new Mat();
        Mat blurFrame = new Mat();
        if (!currentFrame.empty()) {
            Imgproc.resize(currentFrame, blurFrame, new Size(currentFrame.width() / 2, currentFrame.height() / 2));
            Imgproc.GaussianBlur(blurFrame, blurFrame, gaussianFilterSize, 0);
            frame = vibe.foregroundMask(blurFrame);
        }
        return frame;
    }

    boolean isBackgroundSet = false;

    public Mat getDiffDetector() {
        return diffMotionDetector.getDiffDetector(currentFrame);
    }


    public Mat getGaussianBlur() {
        Mat blurFrame = new Mat();
        if (!currentFrame.empty()) {
            //Imgproc.GaussianBlur(currentFrame, blurFrame, gaussianFilterSize, 0);
            pMOG2.getBackgroundImage(blurFrame);
        }
        return blurFrame;
    }

    public Mat getCanny() {
        //mat gray image holder
        Mat imageGray = new Mat();
        //mat canny image
        Mat imageCny = new Mat();
        if (!currentFrame.empty()) {
            //Convert the image in to gray image single channel image
            Imgproc.cvtColor(currentFrame, imageGray, Imgproc.COLOR_BGR2GRAY);
            //Canny Edge Detection
            Imgproc.Canny(imageGray, imageCny, 10, 100, 3, true);
        }
        return imageCny;
    }

    public void setHistoryGMM(int historyGMM) {
        log("Change GMM History to:" + historyGMM);
        pMOG2.setHistory(historyGMM);
    }

    public void setHistoryKNN(int historyKNN) {
        log("Change KNN History to:" + historyKNN);
        pknn.setHistory(historyKNN);
    }


    public void setGaussianFilterSize(int size) {
        int validSize = (size % 2) != 0 ? size : size - 1;
        log("Change Gaussian filter size to:" + validSize);
        gaussianFilterSize = new Size(validSize, validSize);
    }

    public void setKDEThreshole(double threshole) {
        log("Change KDE Threshold to:" + threshole);
        kde.setThreshold(threshole);
    }

    public void setVibeThreshole(double threshole) {
        log("Change Vibe Threshold to:" + threshole);
        vibe.model.setMatchingThreshold(threshole);
    }


    private void log(Object o) {
        System.out.println(o);
    }
}
