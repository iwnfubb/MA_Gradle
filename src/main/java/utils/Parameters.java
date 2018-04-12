package utils;

import org.opencv.core.Scalar;

public class Parameters {
    public static int movementMaximum = 100;  //amount to move to still be the same person
    public static int movementMinimum = 3;   //minimum amount to move to not trigger alarm
    public static int badLimit = 15;     //number of frames after the alarm is triggered

    public static Scalar color_red = new Scalar(0, 0, 255);
    public static Scalar color_blue = new Scalar(255);
    public static Scalar color_green = new Scalar(0, 255, 0);
    public static Scalar color_black = new Scalar(0, 0, 0);
    public static Scalar color_white = new Scalar(255, 255, 255);
    public static Scalar color_gray = new Scalar(126, 126, 126);
}
