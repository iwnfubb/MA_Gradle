package utils;

import javafx.application.Platform;
import javafx.beans.property.ObjectProperty;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.image.Image;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

public final class Utils {
    /**
     * Convert a Mat object (OpenCV) in the corresponding Image for JavaFX
     *
     * @param frame the {@link Mat} representing the current frame
     * @return the {@link Image} to show
     */

    public static Image mat2Image(Mat frame) {
        try {
            return SwingFXUtils.toFXImage(matToBufferedImage(frame), null);
        } catch (Exception e) {
            System.err.println("Cannot convert the Mat object: " + e);
            return null;
        }
    }

    /**
     * Generic method for putting element running on a non-JavaFX thread on the
     * JavaFX thread, to properly update the UI
     *
     * @param property a {@link ObjectProperty}
     * @param value    the value to set for the given {@link ObjectProperty}
     */
    public static <T> void onFXThread(final ObjectProperty<T> property, final T value) {
        Platform.runLater(() -> {
            property.set(value);
        });
    }

    /**
     * @param original the {@link Mat} object in BGR or grayscale
     * @return the corresponding {@link BufferedImage}
     */
    private static BufferedImage matToBufferedImage(Mat original) {
        // init
        BufferedImage image = null;
        int width = original.width(), height = original.height(), channels = original.channels();
        byte[] sourcePixels = new byte[width * height * channels];
        original.get(0, 0, sourcePixels);

        if (original.channels() > 1) {
            image = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
        } else {
            image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        }
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(sourcePixels, 0, targetPixels, 0, sourcePixels.length);

        return image;
    }


    public static double calculateArea(MatOfRect rect) {
        return rect.get(0, 0)[2] * rect.get(0, 0)[3];
    }

    public static double calculateArea(Rect rect) {
        return rect.width * rect.height;
    }


    public static boolean similarArea(MatOfRect rect1, MatOfRect rect2) {
        double a1 = calculateArea(rect1);
        double a2 = calculateArea(rect2);
        if (a1 < a2 && a2 - a1 < a1) {
            return true;
        }
        if (a2 < a1 && a1 - a2 < a2) {
            return true;
        }
        return false;
    }

    public static boolean similarArea(Rect rect1, Rect rect2) {
        double a1 = calculateArea(rect1);
        double a2 = calculateArea(rect2);
        if (a1 < a2 && a2 - a1 < a1) {
            return true;
        }
        if (a2 < a1 && a1 - a2 < a2) {
            return true;
        }
        return false;
    }


    public static boolean overlaps(MatOfRect rect1, MatOfRect rect2) {
        double x1 = rect1.get(0, 0)[0];
        double y1 = rect1.get(0, 0)[1];
        double w1 = rect1.get(0, 0)[2];
        double h1 = rect1.get(0, 0)[3];
        double x2 = rect2.get(0, 0)[0];
        double y2 = rect2.get(0, 0)[1];
        double w2 = rect2.get(0, 0)[2];
        double h2 = rect2.get(0, 0)[3];
        return x1 < x2 + w2 && x1 + w1 > x2
                && y1 < y2 + h2 && y1 + h1 > y2;
    }

    public static boolean overlaps(Rect rect1, Rect rect2) {
        return rect1.x < rect2.x + rect2.width && rect1.x + rect1.width > rect2.x
                && rect1.y < rect2.y + rect2.height && rect1.y + rect1.height > rect2.y;
    }


    public static double euclideandistance(Rect rect1, Rect rect2) {
        double centerX1 = rect1.x + rect1.width / 2;
        double centerY1 = rect1.y + rect1.height / 2;
        double centerX2 = rect2.x + rect2.width / 2;
        double centerY2 = rect2.y + rect2.height / 2;
        return Math.sqrt(Math.pow(centerX1 - centerX2, 2) + Math.pow(centerY1 - centerY2, 2));
    }


    public static Rect convertDoubleToRect(double[] bestRect) {
        return new Rect(bestRect);
    }

    public static Rect2d convertRectToRect2d(Rect r) {
        return new Rect2d(r.x, r.y, r.width, r.height);
    }


    public static Rect convertRect2dToRect(Rect2d r) {
        return new Rect((int) r.x, (int) r.y, (int) r.width, (int) r.height);
    }

    public static MatOfRect convertDoubleToMatOfRect(double[] doubles) {
        MatOfRect matOfRect = new MatOfRect();
        matOfRect.fromArray(convertDoubleToRect(doubles));
        return matOfRect;
    }

    public static MatOfRect convertRectToMatOfRect(Rect rect) {
        MatOfRect matOfRect = new MatOfRect();
        matOfRect.fromArray(rect);
        return matOfRect;
    }


    public static MatOfRect convertRect2dToMatOfRect(Rect2d rect2d) {
        MatOfRect matOfRect = new MatOfRect();
        matOfRect.fromArray(convertRect2dToRect(rect2d));
        return matOfRect;
    }


}