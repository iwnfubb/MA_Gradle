package algorithms;


import org.opencv.core.Rect;
import utils.Utils;

public class MovingDetector {
    public static boolean isObjectMoving(Rect rect1, Rect rect2) {
        double euclideandistance = Utils.euclideandistance(rect1, rect2);
        System.out.println("Moving distance: " + euclideandistance);
        return euclideandistance > 1.5;
    }
}
