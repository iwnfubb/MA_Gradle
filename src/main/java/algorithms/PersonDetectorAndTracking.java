package algorithms;

import net.sourceforge.jFuzzyLogic.optimization.Parameter;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import utils.Parameters;
import utils.Utils;

import java.util.List;

public class PersonDetectorAndTracking {
    boolean startTracking = false;
    PostureDetector postureDetector;
    MovingDetector movingDetector;

    DiffMotionDetector diffMotionDetector;

    FuzzyModel fuzzyModel;

    public PersonDetectorAndTracking() {
        postureDetector = new PostureDetector();
        this.movingDetector = new MovingDetector();
        diffMotionDetector = new DiffMotionDetector();
        fuzzyModel = new FuzzyModel();
    }

    public void startTracking() {
        startTracking = true;
    }

    public void stopTracking() {
        startTracking = false;
    }

    public boolean isTracking() {
        return startTracking;
    }

    public Mat[] detection(Mat input) {
        /*Using Diff Detector*/
        Mat copyOfInput = new Mat();
        input.copyTo(copyOfInput);
        Mat diff_mark = diffMotionDetector.getDiffDetector(copyOfInput);
        Mat binary_mat = diffMotionDetector.thresholdMat;

        diffMotionDetector.personsList.tick();

        Mat imageROI = new Mat();
        if (diffMotionDetector.history.size() != 0) {
            Rect current_rect = diffMotionDetector.history.get(diffMotionDetector.history.size() - 1);
            imageROI = new Mat(binary_mat, current_rect);
        } else {
            binary_mat.copyTo(imageROI);
        }

        Mat status = new Mat(input.size(), CvType.CV_8UC3, Scalar.all(126));
        int posture = postureDetector.detect(imageROI);
        String postureInString = postureDetector.getStatusInString(posture);

        if (diffMotionDetector.history.size() != 0) {
            Rect current_rect = diffMotionDetector.history.get(diffMotionDetector.history.size() - 1);
            Point center = Utils.getCenter(current_rect);
            Person person = diffMotionDetector.personsList.addPerson(current_rect);
            String text = "Moving";
            if (person.lastmoveTime != 0) {
                text = "Not Moving";
            }
            double[] evaluate = fuzzyModel.double_evaluate(posture, 2, center.x, center.y);
            Scalar color = Parameters.color_red;
            person.posture = postureInString;
            person.bad_prediction = evaluate[1];
            person.good_prediction = evaluate[2];
            if (person.alert == 1) {
                Imgproc.line(diff_mark, new Point(person.rect.x, person.rect.y),
                        new Point(person.rect.x + person.rect.width, person.rect.y + person.rect.height),
                        color, 2);
                Imgproc.line(diff_mark, new Point(person.rect.x + person.rect.width, person.rect.y),
                        new Point(person.rect.x, person.rect.y + person.rect.height),
                        color, 2);
            }
            Imgproc.putText(diff_mark, person.getID() + " : " + person.lastmoveTime,
                    new Point(person.rect.x, person.rect.y - 20),
                    Core.FONT_HERSHEY_SIMPLEX, 2, color, 2);

            for (Person p : diffMotionDetector.personsList.persons) {
                drawImageWithRect(diff_mark, p.rect, Parameters.color_blue);
            }

            drawImageWithRect(diff_mark, current_rect, Parameters.color_green);
            writeInfo(status, center.toString(), postureInString, text, evaluate);
        }

        Mat foregroundDisplay = new Mat(input.size(), CvType.CV_8UC1, Scalar.all(126));
        Utils.rescaleImageToDisplay(imageROI, input.width(), input.height());
        imageROI.copyTo(foregroundDisplay.colRange(0, imageROI.cols()).rowRange(0, imageROI.rows()));
        return new Mat[]{input, diff_mark, diffMotionDetector.background_gray, binary_mat, foregroundDisplay, status};
    }


    private void writeInfo(Mat mat, String position, String posture, String moving, double[] evaluation) {
        int fontScale = 2;
        Imgproc.putText(mat, position, new Point(mat.width() / 10, mat.height() / 7),
                0, fontScale, Parameters.color_black, 2);
        Imgproc.putText(mat, posture, new Point(mat.width() / 10, mat.height() / 7 * 2),
                0, fontScale, Parameters.color_blue, 2);
        Scalar movingTextColor = Parameters.color_green;
        if (!moving.equals("Moving")) {
            movingTextColor = Parameters.color_red;
        }
        Imgproc.putText(mat, moving, new Point(mat.width() / 10, mat.height() / 7 * 3),
                0, fontScale, movingTextColor, 2);
        Scalar statusColor = Parameters.color_black;
        Imgproc.putText(mat, "Value: " + evaluation[0], new Point(mat.width() / 10, mat.height() / 7 * 4),
                0, fontScale, statusColor, 2);
        Imgproc.putText(mat, "Bad:" + evaluation[1], new Point(mat.width() / 10, mat.height() / 7 * 5),
                0, fontScale, statusColor, 2);
        Imgproc.putText(mat, "Good:" + evaluation[2], new Point(mat.width() / 10, mat.height() / 7 * 6),
                0, fontScale, statusColor, 2);
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