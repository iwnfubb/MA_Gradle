package algorithms;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class BinaryMaskAnalyser {
    public static int returnNumberOfContours(Mat mask) {
        if (mask.empty()) {
            return 0;
        }
        Mat result = new Mat();
        mask.copyTo(result);
        if (result.channels() == 3) {
            Imgproc.cvtColor(result, result, Imgproc.COLOR_BGR2GRAY);
        }
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(result, contours, hierarchy, 1, 2);
        if (hierarchy.empty()) {
            return 0;
        } else {
            return hierarchy.rows();
        }
    }

    public static Rect returnMaxAreaRectangle(Mat mask) {
        if (mask.empty()) {
            return null;
        }
        Mat result = new Mat();
        mask.copyTo(result);
        if (result.channels() == 3) {
            Imgproc.cvtColor(result, result, Imgproc.COLOR_BGR2GRAY);
        }
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(result, contours, hierarchy, 1, 2);
        double[] area_array = new double[contours.size()];
        int counter = 0;
        for (MatOfPoint cnt : contours) {
            area_array[counter] = Imgproc.contourArea(cnt);
            counter++;
        }

        if (area_array.length == 0) {
            return null;
        }

        int max_area_index = getIndexOfLargest(area_array);
        MatOfPoint cnt = contours.get(max_area_index);
        return Imgproc.boundingRect(cnt);
    }

    private static int getIndexOfLargest(double[] array) {
        if (array == null || array.length == 0) return -1; // null or empty

        int largest = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[largest]) largest = i;
        }
        return largest; // position of the first largest found
    }
}
