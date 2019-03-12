package imageprocess;

import algorithms.PersonDetectorAndTracking;
import net.sf.javaml.clustering.OPTICS;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.core.SparseInstance;
import org.opencv.core.CvType;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.EM;
import org.opencv.videoio.VideoCapture;
import org.opencv.xfeatures2d.SURF;
import utils.Parameters;

import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.opencv.video.Video.calcOpticalFlowFarneback;

public class ImageProcessObjectTracking {
    private static Logger LOGGER = Logger.getLogger(ImageProcessObjectTracking.class.getName());
    private static PersonDetectorAndTracking personDetectorAndTracking;
    private VideoCapture capture;
    private Size gaussianFilterSize = (new Size(3, 3));
    private SURF surf;
    private Mat prevGray = new Mat();

    public ImageProcessObjectTracking(VideoCapture capture) {
        this.capture = capture;
        this.surf = SURF.create();
        this.surf.setUpright(false);
        this.surf.setExtended(true);
        personDetectorAndTracking = new PersonDetectorAndTracking();
    }


    public Mat getOriginalFrame() {
        Mat currentFrame = new Mat();
        if (this.capture.isOpened()) {
            try {
                this.capture.read(currentFrame);
            } catch (Exception e) {
                LOGGER.info("Exception during the image elaboration: " + e);
            }
        }
        return currentFrame;
    }


    public MatOfKeyPoint getSURFKeyPoint(Mat input, Mat mask) {
        MatOfKeyPoint keyPointVector = new MatOfKeyPoint();
        surf.detect(input, keyPointVector, mask);
        return keyPointVector;
    }


    public Mat getGaussianBlur(Mat input) {
        Mat blurFrame = new Mat();
        if (!input.empty()) {
            Imgproc.GaussianBlur(input, blurFrame, gaussianFilterSize, 0);
        }
        return blurFrame;
    }


    public void setGaussianFilterSize(int size) {
        int validSize = (size % 2) != 0 ? size : size - 1;
        LOGGER.log(Level.INFO, "Change Gaussian filter size to: {0}", validSize);
        gaussianFilterSize = new Size(validSize, validSize);
    }


    public void setHessianThreshold(int value) {
        LOGGER.log(Level.INFO, "Change Hessian Threshold to: {0}", value);
        surf.setHessianThreshold(value);
    }


    public void setNOctaveLayer(int value) {
        LOGGER.log(Level.INFO, "Change NOctave Layer to: {0} ", value);
        surf.setNOctaveLayers(value);
    }


    public Mat opticalFLow(Mat input) {
        Mat img = new Mat();
        Mat copyOfOriginal = new Mat();
        Mat flow = new Mat();
        input.copyTo(img);
        input.copyTo(copyOfOriginal);
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2GRAY);

        if (!prevGray.empty()) {
            calcOpticalFlowFarneback(prevGray, img, flow, 0.4, 1, 12, 2, 8, 1.5, 0);
            for (int y = 0; y < copyOfOriginal.rows(); y += 10) {
                for (int x = 0; x < copyOfOriginal.cols(); x += 10) {
                    double flowatx = flow.get(y, x)[0] * 10;
                    double flowaty = flow.get(y, x)[1] * 10;
                    Imgproc.line(copyOfOriginal,
                            new Point(x, y),
                            new Point(Math.round(x + flowatx), Math.round(y + flowaty)),
                            Parameters.color_green);
                    Imgproc.circle(copyOfOriginal,
                            new Point(x, y),
                            2,
                            Parameters.color_black, -2, 4, 0);
                }
            }
            img.copyTo(prevGray);
        } else {
            img.copyTo(prevGray);
        }
        return flow;
    }


    public Mat drawOpticalFlowToImage(Mat input, Mat flow) {
        Mat copyOfOriginal = new Mat();
        input.copyTo(copyOfOriginal);

        if (!flow.empty()) {
            for (int y = 0; y < copyOfOriginal.rows(); y += 10) {
                for (int x = 0; x < copyOfOriginal.cols(); x += 10) {
                    double flowatx = flow.get(y, x)[0] * 10;
                    double flowaty = flow.get(y, x)[1] * 10;
                    Imgproc.line(copyOfOriginal,
                            new Point(x, y),
                            new Point(Math.round(x + flowatx), Math.round(y + flowaty)),
                            Parameters.color_green);
                    Imgproc.circle(copyOfOriginal,
                            new Point(x, y),
                            2,
                            Parameters.color_black, -2, 4, 0);
                }
            }
        }
        return copyOfOriginal;
    }


    public Mat clusteringCoordinateDBSCAN(Mat input, MatOfKeyPoint surfKeyPoint, double eps, int minP) {
        LOGGER.info("Starting Clustering Position My_DBSCAN ...");
        long startTime = System.currentTimeMillis();
        Mat copyOfOriginal = new Mat();
        input.copyTo(copyOfOriginal);
        Dataset data = new DefaultDataset();
        List<KeyPoint> keyPoints = surfKeyPoint.toList();
        for (int i = 0; i < keyPoints.size(); i++) {
            KeyPoint keyPoint = keyPoints.get(i);
            double x = keyPoint.pt.x;
            double y = keyPoint.pt.y;
            //Create instance with 2 Attributes
            Instance instance = new SparseInstance(2);
            instance.put(1, x);
            instance.put(2, y);
            data.add(instance);
        }
        LOGGER.info("Done1");

        //===== OPTIC =====
        OPTICS optics = new OPTICS(eps, minP);
        Dataset[] cluster = optics.cluster(data);

        //===== My_DBSCAN =====
        //DensityBasedSpatialClustering dbscan = new DensityBasedSpatialClustering(eps, minP);
        //Dataset[] cluster = dbscan.cluster(data);

        LOGGER.info("Done2");
        for (int i = 0; i < cluster.length; i++) {
            for (int index = 0; index < cluster[i].size(); index++) {
                Instance instance = cluster[i].get(index);
                Scalar scalar;
                if (i == 0) {
                    scalar = Parameters.color_green;
                } else if (i == 1) {
                    scalar = Parameters.color_red;
                } else {
                    scalar = Parameters.color_blue;
                }
                Imgproc.circle(copyOfOriginal,
                        new Point((int) instance.value(1), (int) instance.value(2)),
                        5,
                        scalar, -5, 4, 0);

            }
        }
        LOGGER.info("Done3");
        LOGGER.log(Level.INFO, "Clustering Time: {0} ", (System.currentTimeMillis() - startTime));
        return copyOfOriginal;
    }


    public Mat clusteringCoordinateGMM(Mat input, MatOfKeyPoint surfKeyPoint) {
        LOGGER.info("Starting Clustering Position GMM...");
        long startTime = System.currentTimeMillis();
        List<KeyPoint> keyPoints = surfKeyPoint.toList();
        Mat labels = new Mat();
        Mat probs = new Mat();
        Mat copyOfOriginal = new Mat();
        input.copyTo(copyOfOriginal);
        float width = copyOfOriginal.width();
        float height = copyOfOriginal.height();
        Mat samples = new Mat(new Size(2, keyPoints.size()), CvType.CV_64FC1);
        for (int i = 0; i < keyPoints.size(); i++) {
            KeyPoint keyPoint = keyPoints.get(i);
            double x = keyPoint.pt.x;
            double y = keyPoint.pt.y;
            samples.put(i, 0, (x / width));
            samples.put(i, 1, (y / height));
        }
        LOGGER.info("Done1");
        EM em = EM.create();
        em.setClustersNumber(3);
        em.setTermCriteria(new TermCriteria(TermCriteria.COUNT, 100, 1));
        em.trainEM(samples, new Mat(), labels, probs);
        LOGGER.info("Done2");

        for (int i = 0; i < keyPoints.size(); i++) {
            KeyPoint keyPoint = keyPoints.get(i);
            Scalar scalar;
            if (labels.get(i, 0)[0] == 0) {
                scalar = Parameters.color_blue;
            } else if (labels.get(i, 0)[0] == 1) {
                scalar = Parameters.color_green;
            } else {
                scalar = Parameters.color_red;
            }
            Imgproc.circle(copyOfOriginal,
                    new Point((int) keyPoint.pt.x, (int) keyPoint.pt.y),
                    5,
                    scalar, -5, 4, 0);
        }
        LOGGER.info("Done3");
        LOGGER.log(Level.INFO, "Clustering Time: {0}", (System.currentTimeMillis() - startTime));
        return copyOfOriginal;
    }


    public Mat[] personDetector(Mat input) {
        return personDetectorAndTracking.detection_KNN(input);
    }

    public boolean isPersonTracking() {
        return personDetectorAndTracking.isTracking();
    }

    public void startPersonTracking() {
        personDetectorAndTracking.startTracking();
    }

    public void stopPersonTracking() {
        personDetectorAndTracking.stopTracking();
    }

    public int getFrameNumber() {
        return personDetectorAndTracking.frame_number;
    }

    public void setFrameNumber(final int frameNumber) {
        personDetectorAndTracking.frame_number = frameNumber;
    }


    public long getProcessTime() {
        return personDetectorAndTracking.processTime;
    }

    public boolean isBackgroundSet() {
        return personDetectorAndTracking.diffMotionDetector.isBackgroundSet;
    }

    public void iniBackground(final boolean value) {
        personDetectorAndTracking.diffMotionDetector.isBackgroundSet = value;
    }

}
