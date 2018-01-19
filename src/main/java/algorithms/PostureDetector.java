package algorithms;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import utils.AverageHistogram;

import javax.swing.plaf.ListUI;
import java.awt.*;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;

public class PostureDetector {
    AverageHistogram averageHistogram;
    ArrayList<Double> last_vertical_histogram = new ArrayList<>();
    ArrayList<Double> last_horizontal_histogram = new ArrayList<>();
    int last_index = -1;
    public boolean falsePositive = false;
    public double fp_threshold = -2.5;

    public PostureDetector() {
        averageHistogram = new AverageHistogram();
    }

    public int detect(Mat input) {
        falsePositive = false;
        if (input.width() < input.height()) {
            Size original_size = input.size();
            double ratio = original_size.height / original_size.width;
            double new_height = 129;
            double new_width = new_height / ratio;
            Mat resizeImage = new Mat();
            Imgproc.resize(input, resizeImage, new Size(new_width, new_height));
            //Imgproc.resize(input, resizeImage, new Size(129, 129));
            ArrayList<Double> vertical_histogram = new ArrayList<>();
            int counter = 0;
            for (int i = 0; i < resizeImage.rows(); i++) {
                if (i > (new_height / 2 - new_width / 2 + 1) && i < (new_height / 2 + new_width / 2 - 1)) {
                    vertical_histogram.add(Core.sumElems(resizeImage.col(counter)).val[0] / 255);
                    counter++;
                } else {
                    vertical_histogram.add(0.0d);
                }

            }

            ArrayList<Double> horizontal_histogram = new ArrayList<>();
            for (int i = 0; i < resizeImage.rows(); i++) {
                horizontal_histogram.add(Core.sumElems(resizeImage.row(i)).val[0] / 255);
            }

            double[] scores = scoreCalculation(vertical_histogram, horizontal_histogram);
            log("## score standing" + scores[0]);
            log("## score laying" + scores[1]);
            log("## score bending" + scores[2]);
            log("## score sitting" + scores[3]);
            ArrayList<Double> temp = new ArrayList<>();
            temp.add(scores[0]);
            temp.add(scores[1]);
            temp.add(scores[2]);
            temp.add(scores[3]);
            Double max = Collections.max(temp);
            int index = temp.indexOf(max);

            double diff = compareWithLastHistogram(vertical_histogram, horizontal_histogram);
            log("## dif :" + diff);
            last_vertical_histogram = vertical_histogram;
            last_horizontal_histogram = horizontal_histogram;
            if (last_index != -1 && last_index != index) {
                if (diff < fp_threshold) {
                    falsePositive = true;
                    log("##### diff change  :" + diff);
                }
            }

            last_index = index;
            return index;
        } else {
            Size original_size = input.size();
            double ratio = original_size.height / original_size.width;
            double new_width = 129;
            double new_height = new_width * ratio;
            Mat resizeImage = new Mat();
            Imgproc.resize(input, resizeImage, new Size(new_width, new_height));
            //Imgproc.resize(input, resizeImage, new Size(129, 129));
            ArrayList<Double> vertical_histogram = new ArrayList<>();
            for (int i = 0; i < resizeImage.cols(); i++) {
                vertical_histogram.add(Core.sumElems(resizeImage.col(i)).val[0] / 126);
            }

            int counter = 0;
            ArrayList<Double> horizontal_histogram = new ArrayList<>();
            for (int i = 0; i < resizeImage.cols(); i++) {
                if (i > (new_width / 2 - new_height / 2 + 1) && i < (new_width / 2 + new_height / 2 - 1)) {
                    horizontal_histogram.add(Core.sumElems(resizeImage.row(counter)).val[0] / 255);
                    counter++;
                } else {
                    horizontal_histogram.add(0.0d);
                }
            }

            double[] scores = scoreCalculation(vertical_histogram, horizontal_histogram);
            log("## score standing" + scores[0]);
            log("## score laying" + scores[1]);
            log("## score bending" + scores[2]);
            log("## score sitting" + scores[3]);
            ArrayList<Double> temp = new ArrayList<>();
            temp.add(scores[0]);
            temp.add(scores[1]);
            temp.add(scores[2]);
            temp.add(scores[3]);
            Double max = Collections.max(temp);
            int index = temp.indexOf(max);

            double diff = compareWithLastHistogram(vertical_histogram, horizontal_histogram);
            log("## dif :" + diff);
            last_vertical_histogram = vertical_histogram;
            last_horizontal_histogram = horizontal_histogram;
            if (last_index != -1 && last_index != index) {
                if (diff < fp_threshold) {
                    falsePositive = true;
                    log("##### diff change  :" + diff);
                }
            }

            last_index = index;
            return index;
        }

    }

    private double[] scoreCalculation(ArrayList<Double> vertical_histogram, ArrayList<Double> horizontal_histogram) {
        double scoreStanding = calculateMinusLogLikelihood(vertical_histogram, horizontal_histogram,
                averageHistogram.vertical_standing, averageHistogram.horizontal_standing);
        double scoreLaying =
                Math.max(
                        calculateMinusLogLikelihood(vertical_histogram, horizontal_histogram,
                                averageHistogram.horizontal_standing, averageHistogram.vertical_standing),
                        calculateMinusLogLikelihood(vertical_histogram, horizontal_histogram,
                                averageHistogram.vertical_laying, averageHistogram.horizontal_laying));
        double scoreBending = calculateMinusLogLikelihood(vertical_histogram, horizontal_histogram,
                averageHistogram.vertical_bending, averageHistogram.horizontal_bending);
        double scoreSitting = calculateMinusLogLikelihood(vertical_histogram, horizontal_histogram,
                averageHistogram.vertical_sitting, averageHistogram.horizontal_sitting);
        return new double[]{scoreStanding, scoreLaying, scoreBending, scoreSitting};
    }


    private double calculateMinusLogLikelihood(ArrayList<Double> vertical_his, ArrayList<Double> horizontal_his,
                                               ArrayList<Double> average_vertical_his, ArrayList<Double> average_horizontal_his) {
        double sum = 0;
        for (int h = 0; h < average_horizontal_his.size(); h++) {
            sum += Math.abs(average_horizontal_his.get(h) - horizontal_his.get(h));
        }
        for (int v = 0; v < average_vertical_his.size(); v++) {
            sum += Math.abs(average_vertical_his.get(v) - vertical_his.get(v));
        }
        return -Math.log10(sum);
    }

    public String getStatusInString(int index) {
        String str = "";
        if (index == 0) {
            str = "Standing";
        }
        if (index == 1) {
            str = "Laying";
        }
        if (index == 2) {
            str = "Bending";
        }
        if (index == 3) {
            str = "Sitting";
        }
        return str;
    }

    private double compareWithLastHistogram(ArrayList<Double> vertical_histogram, ArrayList<Double> horizontal_histogram) {
        if (last_horizontal_histogram.size() == 0 || last_vertical_histogram.size() == 0) {
            return 0.0;
        }
        return calculateMinusLogLikelihood(vertical_histogram, horizontal_histogram,
                last_vertical_histogram, last_horizontal_histogram);
    }

    private void log(Object o) {
        System.out.println(o);
    }

}
