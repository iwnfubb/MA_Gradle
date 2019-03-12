package utils;

import org.opencv.core.Scalar;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Parameters {
    public static int movementMaximum = 100;  //amount to move to still be the same person
    public static int movementMinimum = 3;   //minimum amount to move to not trigger alarm
    public static int badLimit = 15;     //number of frames after the alarm is triggered
    public static double badValue = 0.8;     //bad value

    public static Scalar color_red = new Scalar(0, 0, 255);
    public static Scalar color_blue = new Scalar(255);
    public static Scalar color_green = new Scalar(0, 255, 0);
    public static Scalar color_black = new Scalar(0, 0, 0);
    public static Scalar color_white = new Scalar(255, 255, 255);
    public static Scalar color_gray = new Scalar(126, 126, 126);
    private static Double[] roc = new Double[]{0.8};
    public static List<Double> roc_parameters = Arrays.asList(roc);

}
