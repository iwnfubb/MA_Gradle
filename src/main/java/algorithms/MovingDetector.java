package algorithms;


import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractorKNN;
import org.opencv.video.Video;
import utils.Utils;

public class MovingDetector {
    BackgroundSubtractorKNN bgknn;
    //BackgroundSubtractorMOG2 bgmog2;

    public double backgroundDensity = 0;
    public double[] bestRect;
    public Mat fgmaskClosed = new Mat();


    public MovingDetector() {
        this.bgknn = Video.createBackgroundSubtractorKNN();
        bgknn.setHistory(200);
    }


    public void calculateMostSalientForegroundObject(Mat input) {
        backgroundDensity = 0;
        Mat mask = new Mat();
        bgknn.apply(input, mask, 0.1);

        fgmaskClosed = new Mat();
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(20, 15));
        Imgproc.morphologyEx(mask, fgmaskClosed, Imgproc.MORPH_CLOSE, kernel);

        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();
        int connectivity = 8;
        Imgproc.connectedComponentsWithStats(fgmaskClosed, labels, stats, centroids,
                connectivity, CvType.CV_32S);


        Mat kernelErode = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.erode(fgmaskClosed, fgmaskClosed, kernelErode);
        Mat kernelDalate = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(10, 10));
        Imgproc.dilate(fgmaskClosed, fgmaskClosed, kernelDalate);

        double sum = 0;
        for (int i = 0; i < stats.rows(); i++) {
            sum += stats.get(i, 4)[0];
        }
        backgroundDensity = stats.get(0, 4)[0] / sum;
        bestRect = new double[5];
        if (stats.rows() < 2) {
            bestRect = new double[]{-1, -1, -1, -1, -1};
        } else {
            int mostSalientIndex = 1;
            double max = 0;
            for (int i = 1; i < stats.rows(); i++) {
                if (stats.get(i, 4)[0] > max) {
                    max = stats.get(i, 4)[0];
                    mostSalientIndex = i;
                }
            }
            for (int i = 0; i < bestRect.length; i++)
                bestRect[i] = stats.get(mostSalientIndex, i)[0];
        }
    }


    public static boolean isObjectMoving(Rect rect1, Rect rect2) {
        double euclideandistance = Utils.euclideandistance(rect1, rect2);
        System.out.println("Moving distance: " + euclideandistance);
        return euclideandistance > 1.5;
    }
}
