package algorithms;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import utils.EvaluationValue;
import utils.Parameters;
import utils.Utils;

import java.util.ArrayList;
import java.util.List;

public class PersonDetectorAndTracking {
    boolean startTracking = false;
    PostureDetector postureDetector;
    MovingDetector movingDetector;

    public DiffMotionDetector diffMotionDetector;

    FuzzyModel fuzzyModel;
    public ArrayList<EvaluationValue> list;
    public int frame_number = 0;

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
        long startTime = System.currentTimeMillis();
        diffMotionDetector.personsList.tick();

        for (int i = 0; i < diffMotionDetector.personsList.persons.size(); i++) {
            if (diffMotionDetector.personsList.persons.get(i).forcedDelete) {
                Rect r = diffMotionDetector.personsList.persons.get(i).rect;
                Mat image_gray = new Mat();
                Imgproc.cvtColor(input, image_gray, Imgproc.COLOR_BGR2GRAY);
                Mat imageROI = new Mat(image_gray, r);
                imageROI.copyTo(diffMotionDetector.background_gray.colRange(r.x, r.x + imageROI.cols()).rowRange(r.y, r.y + imageROI.rows()));
                for (int k = 0; k < diffMotionDetector.history.size(); k++) {
                    if (diffMotionDetector.personsList.persons.get(i).exactlySame(diffMotionDetector.history.get(k))) {
                        diffMotionDetector.history.remove(k);
                    }
                }
            }
        }


        /*Using Diff Detector*/
        Mat copyOfInput = new Mat();
        input.copyTo(copyOfInput);
        Mat diff_mark = diffMotionDetector.getDiffDetector(copyOfInput);
        Mat binary_mat = diffMotionDetector.thresholdMat;

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

            if (Utils.overlaps(person.rect, diffMotionDetector.history_knn)) {
                person.forcedDelete = false;
                person.sameBBDetected = 0;
            }

            String moving = "moving";
            if (person.lastmoveTime != 0) {
                moving = "not_moving";
            }
            double[] evaluate = fuzzyModel.double_evaluate(posture, 2, center.x, center.y);
            Scalar color = Parameters.color_red;
            person.posture = postureInString;
            person.bad_prediction = evaluate[1];
            person.good_prediction = evaluate[2];
            if (person.alert) {
                Imgproc.line(diff_mark, new Point(person.rect.x, person.rect.y),
                        new Point(person.rect.x + person.rect.width, person.rect.y + person.rect.height),
                        color, 2);
                Imgproc.line(diff_mark, new Point(person.rect.x + person.rect.width, person.rect.y),
                        new Point(person.rect.x, person.rect.y + person.rect.height),
                        color, 2);
            }
            Imgproc.putText(diff_mark, person.getID() + ":" + person.lastmoveTime,
                    new Point(person.rect.x, person.rect.y - 20),
                    Core.FONT_HERSHEY_SIMPLEX, 1.5, color, 2);
            Imgproc.putText(diff_mark, "ls:" + person.lastseenTime + " sbb:" + person.sameBBDetected,
                    new Point(person.rect.x, person.rect.y - 60),
                    Core.FONT_HERSHEY_SIMPLEX, 1.5, color, 2);

            for (Person p : diffMotionDetector.personsList.persons) {
                drawImageWithRect(diff_mark, p.rect, Parameters.color_blue);
            }

            drawImageWithRect(diff_mark, diffMotionDetector.history_knn, Parameters.color_red);
            drawImageWithRect(diff_mark, current_rect, Parameters.color_green);
            list.get(frame_number).setPerson_there("true");
            list.get(frame_number).setPosture(postureInString);
            list.get(frame_number).setIsMoving(moving);
            if (person.alert) {
                list.get(frame_number).setStatus("not_ok");
            } else {
                list.get(frame_number).setStatus("ok");
            }
            writeInfo(status, center.toString(), postureInString, moving, evaluate);
        }

        Mat foregroundDisplay = new Mat(input.size(), CvType.CV_8UC1, Scalar.all(126));
        Utils.rescaleImageToDisplay(imageROI, input.width(), input.height());
        imageROI.copyTo(foregroundDisplay.colRange(0, imageROI.cols()).rowRange(0, imageROI.rows()));
        System.out.println("##### Time: " + (System.currentTimeMillis() - startTime));
        return new Mat[]{diffMotionDetector.knn_mask, diff_mark, diffMotionDetector.background_gray, binary_mat, foregroundDisplay, status};
    }

    public Mat[] detection_KNN(Mat input) {
        long startTime = System.currentTimeMillis();
        diffMotionDetector.personsList.tick();

        for (int i = 0; i < diffMotionDetector.personsList.persons.size(); i++) {
            if (diffMotionDetector.personsList.persons.get(i).forcedDelete) {
                for (int k = 0; k < diffMotionDetector.history.size(); k++) {
                    if (diffMotionDetector.personsList.persons.get(i).exactlySame(diffMotionDetector.history.get(k))) {
                        diffMotionDetector.history.remove(k);
                    }
                }
            }
        }


        /*Using Diff Detector*/
        Mat copyOfInput = new Mat();
        input.copyTo(copyOfInput);
        Mat diff_mark = diffMotionDetector.getDiffDetector_MOG(copyOfInput);
        Mat binary_mat = diffMotionDetector.thresholdMat;

        Mat imageROI = new Mat();
        if (diffMotionDetector.history.size() != 0 && diffMotionDetector.backgroundDensity >= 0.8) {
            Rect current_rect = diffMotionDetector.history.get(diffMotionDetector.history.size() - 1);
            imageROI = new Mat(binary_mat, current_rect);
        } else {
            binary_mat.copyTo(imageROI);
        }

        Mat status = new Mat(input.size(), CvType.CV_8UC3, Scalar.all(126));
        int posture = postureDetector.detect(imageROI);

        if (diffMotionDetector.history.size() != 0 && diffMotionDetector.backgroundDensity >= 0.8) {
            String postureInString = postureDetector.getStatusInString(posture);
            Rect current_rect = diffMotionDetector.history.get(diffMotionDetector.history.size() - 1);
            Point center = Utils.getCenter(current_rect);
            Person person = diffMotionDetector.personsList.addPerson(current_rect);

            if (Utils.overlaps(person.rect, diffMotionDetector.history_knn)) {
                person.forcedDelete = false;
                person.sameBBDetected = 0;
            }

            String moving = "moving";
            if (person.lastmoveTime != 0) {
                moving = "not_moving";
            }
            double[] evaluate = fuzzyModel.double_evaluate(posture, 2, center.x, center.y);
            Scalar color = Parameters.color_red;
            person.posture = postureInString;
            person.bad_prediction = evaluate[1];
            person.good_prediction = evaluate[2];
            if (person.alert) {
                Imgproc.line(diff_mark, new Point(person.rect.x, person.rect.y),
                        new Point(person.rect.x + person.rect.width, person.rect.y + person.rect.height),
                        color, 2);
                Imgproc.line(diff_mark, new Point(person.rect.x + person.rect.width, person.rect.y),
                        new Point(person.rect.x, person.rect.y + person.rect.height),
                        color, 2);
            }
            Imgproc.putText(diff_mark, person.getID() + ":" + person.lastmoveTime,
                    new Point(person.rect.x, person.rect.y - 20),
                    Core.FONT_HERSHEY_SIMPLEX, 1.5, color, 2);
            Imgproc.putText(diff_mark, "ls:" + person.lastseenTime + " sbb:" + person.sameBBDetected,
                    new Point(person.rect.x, person.rect.y - 60),
                    Core.FONT_HERSHEY_SIMPLEX, 1.5, color, 2);

            for (Person p : diffMotionDetector.personsList.persons) {
                drawImageWithRect(diff_mark, p.rect, Parameters.color_blue);
            }

            drawImageWithRect(diff_mark, diffMotionDetector.history_knn, Parameters.color_red);
            drawImageWithRect(diff_mark, current_rect, Parameters.color_green);
            list.get(frame_number).setPerson_there("true");
            list.get(frame_number).setPosture(postureInString);
            list.get(frame_number).setIsMoving(moving);
            if (person.alert) {
                list.get(frame_number).setStatus("not_ok");
            } else {
                list.get(frame_number).setStatus("ok");
            }
            writeInfo(status, center.toString(), postureInString, moving, evaluate);
        }

        Mat foregroundDisplay = new Mat(input.size(), CvType.CV_8UC1, Scalar.all(126));
        Utils.rescaleImageToDisplay(imageROI, input.width(), input.height());
        imageROI.copyTo(foregroundDisplay.colRange(0, imageROI.cols()).rowRange(0, imageROI.rows()));
        System.out.println("##### Time: " + (System.currentTimeMillis() - startTime));
        return new Mat[]{diffMotionDetector.knn_mask, diff_mark, diffMotionDetector.background_gray, binary_mat, foregroundDisplay, status};
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