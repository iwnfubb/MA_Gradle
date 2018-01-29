package algorithms;

import org.opencv.core.*;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.video.BackgroundSubtractorKNN;
import org.opencv.video.Video;
import utils.Utils;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class PersonDetectorAndTracking {
    boolean startTracking = false;
    HOGDescriptor hog;
    BackgroundSubtractorKNN bgknn;
    //BackgroundSubtractorMOG2 bgmog2;
    PersonTracker tracker;
    FeatureDetector blob;
    PostureDetector postureDetector;
    MovingDetector movingDetector;
    boolean tracking = false;
    Rect last_rect = new Rect();

    //HOG Parameter
    double hitThreshold = 0;
    Size winStride = new Size(4, 4);
    Size padding = new Size(8, 8);
    double scale = 1.05;
    boolean useMeanshiftGrouping = true;
    double finalThreshold = 0;
    DiffMotionDetector diffMotionDetector;

    FuzzyModel fuzzyModel;
    int alarmCounter = 0;


    public PersonDetectorAndTracking() {
        this.hog = new HOGDescriptor();
        MatOfFloat peopleDetector = HOGDescriptor.getDefaultPeopleDetector();
        hog.setSVMDetector(peopleDetector);

        this.bgknn = Video.createBackgroundSubtractorKNN();
        bgknn.setHistory(200);

        //this.bgmog2 = Video.createBackgroundSubtractorMOG2();
        //bgmog2.setHistory(500);
        blob = FeatureDetector.create(FeatureDetector.SIMPLEBLOB);
        blob.read("blob.xml");

        tracker = new PersonTracker();

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


    public Mat[] detect2(Mat input) {
        /**second try with person tracking
         * problem: work only indoor without background change
         * */
        Mat person = new Mat();
        input.copyTo(person);
        MatOfRect foundLocations = new MatOfRect();
        MatOfDouble foundWeights = new MatOfDouble();

        if (!startTracking) {
            movingDetector.calculateMostSalientForegroundObject(input);
            Mat connectedMat = movingDetector.fgmaskClosed;
            Mat imageWithBestRect = new Mat();

            hog.detectMultiScale(person, foundLocations, foundWeights, hitThreshold, winStride, padding, scale,
                    finalThreshold, useMeanshiftGrouping);
            foundLocations = filterPersonArea(foundLocations, foundWeights);
            drawRect(person, foundLocations, new Scalar(0, 0, 255));

            input.copyTo(imageWithBestRect);
            drawImageWithRect(imageWithBestRect, movingDetector.bestRect, new Scalar(255, 0, 0));
            drawImageWithRect(person, movingDetector.bestRect, new Scalar(255, 0, 0));
            Rect r = Utils.convertDoubleToRect(movingDetector.bestRect);
            log("Background backgroundDensity: " + movingDetector.backgroundDensity);

            if (isBestRectDetected(r, foundLocations) != -1 && movingDetector.backgroundDensity > 0.8) {
                Imgproc.cvtColor(input, input, Imgproc.COLOR_RGB2HSV);
                tracker.createTracker(input, r);
                startTracking = true;
            }
            return new Mat[]{person, imageWithBestRect, connectedMat};
        } else {
            movingDetector.calculateMostSalientForegroundObject(input);
            Mat connectedMat = movingDetector.fgmaskClosed;
            Imgproc.cvtColor(person, person, Imgproc.COLOR_RGB2HSV);
            startTracking = tracker.updateTrackingbox(person);

            Mat[] segmentations = new Mat[3];
            boolean isPersonThere = false;
            if (startTracking) {
                Mat imageROI = new Mat(person, tracker.getTrackingBoxAsRect());
                Mat connectedMatROI = new Mat(connectedMat, tracker.getTrackingBoxAsRect());
                segmentations = imageSegmentaion3(imageROI, connectedMatROI);

                //if false positive -> calculate based on last rect
                if (postureDetector.falsePositive) {
                    //Core and
                    tracker.restoreCurrentTrackingBoxFromLast();
                    imageROI = new Mat(person, tracker.getTrackingBoxAsRect());
                    connectedMatROI = new Mat(connectedMat, tracker.getTrackingBoxAsRect());
                    segmentations = imageSegmentaion3(imageROI, connectedMatROI);
                }
                hog.detectMultiScale(imageROI, foundLocations, foundWeights, hitThreshold, winStride, padding, scale,
                        finalThreshold, useMeanshiftGrouping);
                if (foundWeights.rows() != 0) {
                    List<Double> doubles = foundWeights.toList();

                    log("#Recheck ");
                    for (Double d : doubles) {
                        if (d > 0.5) {
                            isPersonThere = true;
                            log("!!! Person detected !!!");
                        }
                    }
                    startTracking = startTracking && isPersonThere;
                }
            }

            if (!startTracking && !isPersonThere) {
                Rect r = Utils.convertDoubleToRect(movingDetector.bestRect);
                tracker.createTracker(input, r);
                startTracking = true;
            }

            drawRect(person, tracker.getTrackingBoxAsMatOfRect(), new Scalar(0, 255, 0));
            boolean objectMoving = MovingDetector.isObjectMoving(tracker.getTrackingBoxAsRect(),
                    tracker.getLast_TrackingBoxAsRect());
            String moving = "";
            if (objectMoving) {
                moving = "Moving";
            }

            Imgproc.putText(person, moving, new Point(30, 30),
                    0, 2, new Scalar(0, 0, 255), 3);
            tracker.saveTrackingBoxToMemory();
            return new Mat[]{person, person, connectedMat,
                    Utils.rescaleImageToDisplay(segmentations[0], input.width(), input.height()),
                    Utils.rescaleImageToDisplay(segmentations[1], input.width(), input.height()),
                    Utils.rescaleImageToDisplay(segmentations[2], input.width(), input.height())};
        }
    }

    public Mat[] detect4(Mat input) {
        /*Using Diff Detector*/
        int fontSCale = 2;
        Mat copyOfInput = new Mat();
        input.copyTo(copyOfInput);
        Mat diff_mark = diffMotionDetector.getDiffDetector(copyOfInput);
        Mat binary_mat = diffMotionDetector.thresholdMat;

        Mat imageROI = new Mat();
        if (diffMotionDetector.history.size() != 0) {
            Rect current_rect = diffMotionDetector.history.get(diffMotionDetector.history.size() - 1);
            imageROI = new Mat(binary_mat, current_rect);
            drawImageWithRect(diff_mark, current_rect, new Scalar(0, 255, 0));
        } else {
            binary_mat.copyTo(imageROI);
        }

        Mat status = new Mat(input.size(), CvType.CV_8UC3, Scalar.all(126));
        int posture = postureDetector.detect(imageROI);
        String postureInString = postureDetector.getStatusInString(posture);
        boolean moving = diffMotionDetector.isObjectMoving;

        Imgproc.putText(status, postureInString, new Point(status.width() / 10, status.height() / 5),
                0, fontSCale, new Scalar(255), 2);
        Scalar movingTextColor = new Scalar(0, 255);
        Scalar alarmTextColor = new Scalar(0, 255);
        String text = "Moving";
        if (!moving) {
            if (postureInString.equals("Laying")) {
                alarmCounter++;
            } else {
                alarmCounter = 0;
            }
            text = "Not Moving";
        }


        if (alarmCounter >= 10) {
            alarmTextColor = new Scalar(0, 0, 255);
        } else {
            alarmTextColor = new Scalar(0, 255, 0);
        }
        Imgproc.putText(status, "ALARM!!!", new Point(status.width() / 10, status.height() / 1.9),
                0, fontSCale, alarmTextColor, 2);


        Imgproc.putText(status, text, new Point(status.width() / 10, status.height() / 3),
                0, fontSCale, movingTextColor, 2);

        if (diffMotionDetector.history.size() != 0) {
            Rect current_rect = diffMotionDetector.history.get(diffMotionDetector.history.size() - 1);
            Point center = Utils.getCenter(current_rect);
            log(center.toString());
            String[] evaluate = fuzzyModel.evaluate(posture, 2, center.x, center.y);
            Scalar statusColor = new Scalar(0, 0, 0);
            Imgproc.putText(status, evaluate[0], new Point(status.width() / 10, status.height() / 2),
                    0, fontSCale, statusColor, 2);
            Imgproc.putText(status, evaluate[1], new Point(status.width() / 10, status.height() / 1.5),
                    0, fontSCale, statusColor, 2);
            Imgproc.putText(status, evaluate[2], new Point(status.width() / 10, status.height() / 1.7),
                    0, fontSCale, statusColor, 2);

        }

        Mat foregroundDisplay = new Mat(input.size(), CvType.CV_8UC1, Scalar.all(126));
        Utils.rescaleImageToDisplay(imageROI, input.width(), input.height());
        imageROI.copyTo(foregroundDisplay.colRange(0, imageROI.cols()).rowRange(0, imageROI.rows()));

        return new Mat[]{input, diff_mark, diffMotionDetector.background_gray, binary_mat, foregroundDisplay, status};
    }

    public Mat[] detect3(Mat input) {
        /**third try with person tracking
         * try to remove HOG , it is not needed
         * track based on moving object
         * compare moving box with tracking box
         * if moving box is problely a person -> create new track
         *
         */
        /**third try with person tracking
         * create a list of tracker (tracking motion and tracking person)
         * after 10 frames if object it not moving then remove them from list
         * if object at least 1 frame moving then track always (or add more chances ???)
         *
         *
         */
        Mat person = new Mat();
        input.copyTo(person);
        MatOfRect foundLocations = new MatOfRect();
        MatOfDouble foundWeights = new MatOfDouble();

        if (!startTracking) {
            movingDetector.calculateMostSalientForegroundObject(input);
            Mat connectedMat = movingDetector.fgmaskClosed;
            Mat imageWithBestRect = new Mat();

            hog.detectMultiScale(person, foundLocations, foundWeights, hitThreshold, winStride, padding, scale,
                    finalThreshold, useMeanshiftGrouping);
            foundLocations = filterPersonArea(foundLocations, foundWeights);
            drawRect(person, foundLocations, new Scalar(0, 0, 255));

            input.copyTo(imageWithBestRect);
            drawImageWithRect(imageWithBestRect, movingDetector.bestRect, new Scalar(255, 0, 0));
            drawImageWithRect(person, movingDetector.bestRect, new Scalar(255, 0, 0));
            Rect r = Utils.convertDoubleToRect(movingDetector.bestRect);
            log("Background backgroundDensity: " + movingDetector.backgroundDensity);

            if (isBestRectDetected(r, foundLocations) != -1 && movingDetector.backgroundDensity > 0.8) {
                tracker.createTracker(input, r);
                startTracking = true;
            }
            return new Mat[]{person, imageWithBestRect, connectedMat};
        } else {
            movingDetector.calculateMostSalientForegroundObject(input);
            Mat connectedMat = movingDetector.fgmaskClosed;
            startTracking = tracker.updateTrackingbox(person);

            Mat[] segmentations = new Mat[3];
            if (startTracking) {
                Mat imageROI = new Mat(person, tracker.getTrackingBoxAsRect());
                Mat connectedMatROI = new Mat(connectedMat, tracker.getTrackingBoxAsRect());
                segmentations = imageSegmentaion3(imageROI, connectedMatROI);
            }

            //if tracking lost or person reappear in frame0
            if (!startTracking ||
                    (Utils.overlaps(tracker.getTrackingBoxAsRect(), Utils.convertDoubleToRect(movingDetector.bestRect)) &&
                            Utils.similarArea(tracker.getTrackingBoxAsRect(), Utils.convertDoubleToRect(movingDetector.bestRect)))) {
                Rect r = Utils.convertDoubleToRect(movingDetector.bestRect);
                Mat imageROI = new Mat(person, r);
                Mat connectedMatROI = new Mat(connectedMat, r);
                segmentations = imageSegmentaion3(imageROI, connectedMatROI);
                tracker.createTracker(input, r);
                startTracking = true;
            }

            drawRect(person, tracker.getTrackingBoxAsMatOfRect(), new Scalar(0, 255, 0));
            drawRect(person, Utils.convertDoubleToMatOfRect(movingDetector.bestRect), new Scalar(255, 0, 0));
            boolean objectMoving = MovingDetector.isObjectMoving(tracker.getTrackingBoxAsRect(),
                    tracker.getLast_TrackingBoxAsRect());
            String moving = "";
            if (objectMoving) {
                moving = "Moving";
            }

            Imgproc.putText(person, moving, new Point(30, 30),
                    0, 2, new Scalar(0, 0, 255), 3);
            tracker.saveTrackingBoxToMemory();
            return new Mat[]{person,
                    person,
                    connectedMat,
                    Utils.rescaleImageToDisplay(segmentations[0], input.width(), input.height()),
                    Utils.rescaleImageToDisplay(segmentations[1], input.width(), input.height()),
                    Utils.rescaleImageToDisplay(segmentations[2], input.width(), input.height())};
        }
    }

    public Mat[] detect(Mat input) {
        /* first try with detection
            problem : tracking is not correct , doesnt follow moving person if a similar object occurs
        */
        Mat person = new Mat();
        input.copyTo(person);
        if (!startTracking) {
            tracking = false;

            Mat imageWithBestRect = new Mat();
            input.copyTo(imageWithBestRect);
            movingDetector.calculateMostSalientForegroundObject(input);
            Mat connectedMat = movingDetector.fgmaskClosed;
            Mat[] segmentations;
            Rect r = new Rect(movingDetector.bestRect);
            MatOfRect bestRect = new MatOfRect();
            bestRect.fromArray(r);
            drawRect(imageWithBestRect, bestRect, new Scalar(255, 0, 0));

            if (movingDetector.backgroundDensity > 0.8) {
                Mat imageROI = new Mat(input, r);
                Mat connectedMatROI = new Mat(connectedMat, r);
                segmentations = imageSegmentaion3(imageROI, connectedMatROI);
                return new Mat[]{imageWithBestRect, connectedMat, imageROI, segmentations[0], segmentations[1], segmentations[2]};
            } else {
                return new Mat[]{imageWithBestRect};
            }
        } else {
            if (!tracking) {
                Mat imageWithBestRect = new Mat();
                input.copyTo(imageWithBestRect);
                movingDetector.calculateMostSalientForegroundObject(input);
                Mat connectedMat = movingDetector.fgmaskClosed;
                Rect r = new Rect(movingDetector.bestRect);
                tracker.createTracker(input, r);
                tracking = true;
            } else {
                tracking = tracker.updateTrackingbox(person);
            }
            drawRect(person, tracker.getTrackingBoxAsMatOfRect(), new Scalar(0, 255, 0));
            return new Mat[]{person};
        }
    }

    private MatOfRect filterPersonArea(MatOfRect foundLocations, MatOfDouble foundWeights) {
        if (foundWeights.rows() != 0) {
            List<Rect> rects = foundLocations.toList();
            List<Double> ws = foundWeights.toList();
            List<Rect> newList = new ArrayList<>();
            log("#Filter");
            for (Double d : ws) {
                if (d > 0.5) {
                    newList.add(rects.get(ws.indexOf(d)));
                    log("!!! Person detected !!!");
                }
            }
            if (newList.size() != 0) {
                foundLocations.fromList(newList);
            } else {
                foundLocations = new MatOfRect();
            }
        }
        return foundLocations;
    }

    private void log(Object o) {
        System.out.println(o);
    }

    private void drawRect(Mat img, MatOfRect matOfRect, Scalar color) {
        List<Rect> rects = matOfRect.toList();
        for (Rect r : rects) {
            Imgproc.rectangle(img, r.tl(), r.br(), color, 5);
        }
    }


    private int isBestRectDetected(Rect bestrect, MatOfRect rects) {
        List<Rect> rectslist = rects.toList();
        for (int i = 0; i < rectslist.size(); i++) {
            if (Utils.isRect1InsideRect2(bestrect, rectslist.get(i)) && bestrect.area() > 0.5 * rectslist.get(i).area()) {
                return i;
            }
        }
        return -1;
    }


    private Mat getBlob(Mat input) {
        Mat result = new Mat();
        input.copyTo(result);
        ColorBlobDetector colorBlob = new ColorBlobDetector();

        Mat regionHsv = new Mat();
        Imgproc.cvtColor(input, regionHsv, Imgproc.COLOR_RGB2HSV_FULL);

        // Calculate average color of touched region
        Scalar mBlobColorHsv = Core.sumElems(regionHsv);
        int pointCount = 0;
        for (int y = 0; y < input.rows(); y++)
            for (int x = 0; x < input.cols(); x++) {
                if (input.get(y, x)[0] == 0 && input.get(y, x)[1] == 0 && input.get(y, x)[2] == 0) {
                    continue;
                } else {
                    pointCount++;
                }
            }
        for (int i = 0; i < mBlobColorHsv.val.length; i++)
            mBlobColorHsv.val[i] /= pointCount;

        colorBlob.setHsvColor(mBlobColorHsv);
        colorBlob.process(input);

        for (int i = 0; i < colorBlob.getContours().size(); i++)
            Imgproc.drawContours(result, colorBlob.getContours(), i, new Scalar(new Random().nextInt(255), new Random().nextInt(255)), -1);


        return result;
    }

    private Mat drawImageWithRect(Mat input, double[] bestRect, Scalar color) {
        Rect r = new Rect(bestRect);
        MatOfRect matOfRect = new MatOfRect();
        matOfRect.fromArray(r);
        drawRect(input, matOfRect, color);
        return input;
    }

    private Mat drawImageWithRect(Mat input, Rect bestRect, Scalar color) {
        MatOfRect matOfRect = new MatOfRect();
        matOfRect.fromArray(bestRect);
        drawRect(input, matOfRect, color);
        return input;
    }


    public Mat[] imageSegmentaion(Mat input) {
        double thresh = 70;
        Mat edges = new Mat();
        //Canny
        log("Canny");
        Utils.convertImageByInvariantFeatures(input).copyTo(input);
        Imgproc.cvtColor(input, edges, Imgproc.COLOR_BGR2GRAY);
        Imgproc.Canny(edges, edges, thresh, thresh * 1.5, 3, true);

        //Mor
        log("morphologyEx");
        Mat fgmaskClosed = new Mat();
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.morphologyEx(edges, fgmaskClosed, Imgproc.MORPH_GRADIENT, kernel);

        //Contours -> Foreground mask
        log("findContours");
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(fgmaskClosed, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        Mat foregroundMask = new Mat(edges.size(), CvType.CV_8UC1, new Scalar(0));
        for (int i = 0; i < contours.size(); i++)
            Imgproc.drawContours(foregroundMask, contours, i, new Scalar(255), -1);

        log("laplacian");
        //Input with foreground mask
        Mat inputWithForeGroundMask = new Mat();
        input.copyTo(inputWithForeGroundMask, foregroundMask);

        //Laplacian filter
        Mat laplacianKernel = new Mat(new Size(3, 3), CvType.CV_32FC1);
        laplacianKernel.put(0, 0,
                new double[]{1, 1, 1,
                        1, -8, 1,
                        1, 1, 1});
        Mat imgLaplacian = new Mat();
        Imgproc.filter2D(inputWithForeGroundMask, imgLaplacian, CvType.CV_32F, kernel);
        Mat sharp = new Mat();
        input.copyTo(sharp);

        input.convertTo(sharp, CvType.CV_32FC1);
        Mat imgResult = new Mat();
        Core.subtract(sharp, imgLaplacian, imgResult);

        imgResult.convertTo(imgResult, CvType.CV_8UC3);
        imgLaplacian.convertTo(imgLaplacian, CvType.CV_8UC3);

        //convert input image to back white
        Mat bw = new Mat();
        Imgproc.cvtColor(input, bw, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(bw, bw, 40, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);

        //Distance transform
        log("distanceTransform");
        Mat dist = new Mat();
        Imgproc.distanceTransform(bw, dist, Imgproc.CV_DIST_L2, 3);
        Core.normalize(dist, dist, 0, 255.0, Core.NORM_MINMAX);

        //morphology operation
        log("dilate");
        Mat dist1 = new Mat();
        dist.copyTo(dist1);
        Imgproc.threshold(dist1, dist1, 0.4 * 255, 1.0 * 255, Imgproc.THRESH_BINARY);
        Mat kernel1 = new Mat(3, 3, CvType.CV_8UC1, Scalar.all(1));
        Imgproc.dilate(dist1, dist1, kernel1);
        //dist1.convertTo(dist1, CvType.CV_8UC1);

        //create seed
        log("findContours");
        Mat dist_8u = new Mat();
        dist1.convertTo(dist_8u, CvType.CV_8U);
        List<MatOfPoint> contours1 = new ArrayList<>();
        Imgproc.findContours(dist_8u, contours1, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        Mat markers = new Mat(dist.size(), CvType.CV_32SC1, Scalar.all(0));

        for (int i = 0; i < contours1.size(); i++) {
            Imgproc.drawContours(markers, contours1, i, Scalar.all(i + 1), -1);
        }

        Imgproc.circle(markers, new Point(5, 5), 3, new Scalar(255, 255, 255), -1);
        //markers*1000


        //Watershed
        log("watershed");
        Imgproc.watershed(input, markers);
        Mat mark = new Mat(markers.size(), CvType.CV_32SC1, Scalar.all(0)); //mark just for test
        Core.bitwise_not(mark, mark);

        List<Scalar> randomColors = new ArrayList<>();
        Random rand = new Random();
        for (int i = 0; i < contours1.size(); i++) {
            randomColors.add(new Scalar(rand.nextInt(255), rand.nextInt(255), rand.nextInt(255)));
        }

        Mat dst = new Mat(markers.size(), CvType.CV_8UC3, Scalar.all(0));

        for (int i = 0; i < markers.rows(); i++)
            for (int j = 0; j < markers.cols(); j++) {
                int index = new Double(markers.get(i, j)[0]).intValue();
                if (index > 0 && index < contours1.size()) {
                    dst.put(i, j, randomColors.get(index).val[0], randomColors.get(index).val[1], randomColors.get(index).val[2]);
                } else {
                    dst.put(i, j, 0, 0, 0);
                }
            }


        log("blob");
        MatOfKeyPoint blobKeyPoint = new MatOfKeyPoint();
        Mat blobImage = new Mat();
        inputWithForeGroundMask.copyTo(blobImage);
        Imgproc.cvtColor(blobImage, blobImage, Imgproc.COLOR_RGB2GRAY);
        blob.detect(blobImage, blobKeyPoint);

        Features2d.drawKeypoints(blobImage, blobKeyPoint, blobImage, new Scalar(0, 0, 255, 0), 4);

        //return new Mat[]{foregroundMask, imgLaplacian, imgResult, bw, dist, dist1, dst};
        //return new Mat[]{foregroundMask, imgLaplacian, imgResult, bw, dist, dist1, dst};
        return new Mat[]{inputWithForeGroundMask, fgmaskClosed, imgLaplacian, dst, blobImage};
    }

    public Mat[] imageSegmentaion2(Mat input) {
        double thresh = 70;
        Mat edges = new Mat();
        //Canny
        log("Canny");
        Mat inputWithinvariantFeatures = new Mat();
        Utils.convertImageByInvariantFeatures(input).copyTo(inputWithinvariantFeatures);
        Imgproc.cvtColor(inputWithinvariantFeatures, edges, Imgproc.COLOR_BGR2GRAY);
        Imgproc.Canny(edges, edges, thresh, thresh * 1.5, 3, true);

        //Mor
        log("morphologyEx");
        Mat fgmaskClosed = new Mat();
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.morphologyEx(edges, fgmaskClosed, Imgproc.MORPH_GRADIENT, kernel);

        //Contours -> Foreground mask
        log("findContours");
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(fgmaskClosed, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        Mat foregroundMask = new Mat(edges.size(), CvType.CV_8UC1, new Scalar(0));
        for (int i = 0; i < contours.size(); i++)
            Imgproc.drawContours(foregroundMask, contours, i, new Scalar(255), -1);

        //find bounding curve
        Mat foregroundMaskWithBorder = new Mat();
        Core.copyMakeBorder(foregroundMask, foregroundMaskWithBorder, 10, 10, 10, 10, Core.BORDER_CONSTANT);
        List<MatOfPoint> boundingContours = new ArrayList<>();
        Imgproc.findContours(foregroundMaskWithBorder, boundingContours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        //hull curve
        Mat boundingCurve = new Mat(foregroundMaskWithBorder.size(), CvType.CV_8UC1, new Scalar(0));
        List<MatOfInt> hullCurve = new ArrayList<>();
        for (int i = 0; i < boundingContours.size(); i++) {
            MatOfInt hull = new MatOfInt();
            Imgproc.drawContours(boundingCurve, boundingContours, i, new Scalar(255), 1);
            Imgproc.convexHull(boundingContours.get(i), hull, true);
            hullCurve.add(hull);
        }
        //draw hull points

        for (int i = 0; i < boundingContours.size(); i++) {
            MatOfInt hull = hullCurve.get(i);
            int index = (int) hull.get(((int) hull.size().height) - 1, 0)[0];
            Point pt, pt0 = new Point(boundingContours.get(i).get(index, 0)[0], boundingContours.get(i).get(index, 0)[1]);
            for (int j = 0; j < hull.size().height - 1; j++) {
                index = (int) hull.get(j, 0)[0];
                pt = new Point(boundingContours.get(i).get(index, 0)[0], boundingContours.get(i).get(index, 0)[1]);
                Imgproc.line(boundingCurve, pt0, pt, new Scalar(255), 1);
                Imgproc.circle(boundingCurve, pt0, 5,
                        new Scalar(255), -5, 4, 0);
                pt0 = pt;
            }
        }

        //Input with foreground mask
        Mat inputWithForeGroundMask = new Mat();
        input.copyTo(inputWithForeGroundMask, foregroundMask);


        log("blob");
        MatOfKeyPoint blobKeyPoint = new MatOfKeyPoint();
        Mat blobImage = new Mat();
        inputWithForeGroundMask.copyTo(blobImage);
        Imgproc.cvtColor(inputWithForeGroundMask, blobImage, Imgproc.COLOR_RGB2GRAY);
        blob.detect(blobImage, blobKeyPoint);


        Features2d.drawKeypoints(blobImage, blobKeyPoint, blobImage, new Scalar(0, 0, 255, 0), 4);
        return new Mat[]{inputWithForeGroundMask, foregroundMask, boundingCurve};
    }


    public Mat[] imageSegmentaion3(Mat input, Mat connectedMatROI) {
        double thresh = 70;
        Mat edges = new Mat();
        //Canny
        log("Canny");
        Mat inputWithinvariantFeatures = new Mat();
        Utils.convertImageByInvariantFeatures(input).copyTo(inputWithinvariantFeatures);
        Imgproc.cvtColor(inputWithinvariantFeatures, edges, Imgproc.COLOR_BGR2GRAY);
        Imgproc.Canny(edges, edges, thresh, thresh * 1.5, 3, true);

        //Mor
        log("morphologyEx");
        Mat fgmaskClosed = new Mat();
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.morphologyEx(edges, fgmaskClosed, Imgproc.MORPH_GRADIENT, kernel);

        //Contours -> Foreground mask
        log("findContours");
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(fgmaskClosed, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        Mat foregroundMask = new Mat(edges.size(), CvType.CV_8UC1, new Scalar(0));
        for (int i = 0; i < contours.size(); i++)
            Imgproc.drawContours(foregroundMask, contours, i, new Scalar(255), -1);

        //Core.bitwise_and(foregroundMask, connectedMatROI, foregroundMask);
        //find bounding curve
        Mat foregroundMaskWithBorder = new Mat();
        Core.copyMakeBorder(foregroundMask, foregroundMaskWithBorder, 10, 10, 10, 10, Core.BORDER_CONSTANT);
        List<MatOfPoint> boundingContours = new ArrayList<>();
        Imgproc.findContours(foregroundMaskWithBorder, boundingContours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        Mat boundingCurve = new Mat(foregroundMaskWithBorder.size(), CvType.CV_8UC1, new Scalar(0));
        for (int i = 0; i < boundingContours.size(); i++) {
            Imgproc.drawContours(boundingCurve, boundingContours, i, new Scalar(255), 1);
        }

        //Input with foreground mask
        Mat inputWithForeGroundMask = new Mat();
        input.copyTo(inputWithForeGroundMask, foregroundMask);


        String posture = postureDetector.getStatusInString(postureDetector.detect(foregroundMaskWithBorder));
        Imgproc.putText(foregroundMaskWithBorder, posture, new Point(10, 10),
                0, 0.5, new Scalar(255), 2);

        return new Mat[]{inputWithForeGroundMask, foregroundMask, foregroundMaskWithBorder};
    }


}