package imageprocess;

import algorithms.Clustering;
import algorithms.PersonDetectorAndTracking;
import algorithms.PostureDetector;
import net.sf.javaml.clustering.OPTICS;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.core.SparseInstance;
import org.opencv.core.*;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.FlannBasedMatcher;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.EM;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.tracking.Tracker;
import org.opencv.tracking.TrackerKCF;
import org.opencv.video.BackgroundSubtractorKNN;
import org.opencv.video.BackgroundSubtractorMOG2;
import org.opencv.video.Video;
import org.opencv.videoio.VideoCapture;
import org.opencv.xfeatures2d.SURF;
import utils.KeyPointsAndFeaturesVector;

import java.util.ArrayList;
import java.util.List;

import static org.opencv.video.Video.calcOpticalFlowFarneback;

public class ImageProcess_ObjectTracking {
    private VideoCapture capture;
    private Size gaussianFilterSize = (new Size(3, 3));
    private SURF surf;
    private Mat prevgray = new Mat();
    private Mat previousFrame = new Mat();
    private Mat previousFrameDescriptors = new Mat();
    private MatOfKeyPoint previousKeyPoint = new MatOfKeyPoint();
    private int frameCounter = 0;
    private KeyPointsAndFeaturesVector backgroundModelTobi;
    private Mat backGroundModel;
    boolean initBackgroundModel = false;
    HOGDescriptor hog;
    BackgroundSubtractorMOG2 mog2;
    BackgroundSubtractorKNN bgknn;
    Tracker tracker;
    FeatureDetector blob;
    PostureDetector postureDetector;
    public PersonDetectorAndTracking personDetectorAndTracking;

    public ImageProcess_ObjectTracking(VideoCapture capture) {
        this.capture = capture;
        this.surf = SURF.create();
        this.surf.setUpright(false);
        this.surf.setExtended(true);

        /*this.hog = new HOGDescriptor(new Size(48, 96),
                new Size(16, 16),
                new Size(8, 8),
                new Size(8, 8),
                9, 1,-1, HOGDescriptor.L2Hys, 1.0, true, HOGDescriptor.DEFAULT_NLEVELS, false);
        */
        this.hog = new HOGDescriptor();
        MatOfFloat peopleDetector = HOGDescriptor.getDefaultPeopleDetector();
        hog.setSVMDetector(peopleDetector);

        this.mog2 = Video.createBackgroundSubtractorMOG2();
        mog2.setHistory(50);

        this.bgknn = Video.createBackgroundSubtractorKNN();
        bgknn.setHistory(50);

        blob = FeatureDetector.create(FeatureDetector.SIMPLEBLOB);
        blob.read("blob.xml");

        createMyTracker();

        postureDetector = new PostureDetector();

        personDetectorAndTracking = new PersonDetectorAndTracking();
    }

    private void createMyTracker() {
        this.tracker = TrackerKCF.create();
    }


    public Mat getOriginalFrame() {
        Mat currentFrame = new Mat();
        if (this.capture.isOpened()) {
            try {
                this.capture.read(currentFrame);
            } catch (Exception e) {
                System.err.println("Exception during the image elaboration: " + e);
            }
        }
        return currentFrame;
    }


    public MatOfKeyPoint getSURFKeyPoint(Mat input, Mat mask) {
        MatOfKeyPoint keyPointVector = new MatOfKeyPoint();
        surf.detect(input, keyPointVector, mask);
        return keyPointVector;
    }


    public Mat getGaussianBlur(Mat input) {
        Mat blurFrame = new Mat();
        if (!input.empty()) {
            Imgproc.GaussianBlur(input, blurFrame, gaussianFilterSize, 0);
        }
        return blurFrame;
    }


    public void setGaussianFilterSize(int size) {
        int validSize = (size % 2) != 0 ? size : size - 1;
        log("Change Gaussian filter size to:" + validSize);
        gaussianFilterSize = new Size(validSize, validSize);
    }


    public void setHessianThreshold(int value) {
        log("Change Hessian Threshold to:" + value);
        surf.setHessianThreshold(value);
    }


    public void setNOctaveLayer(int value) {
        log("Change NOctave Layer to:" + value);
        surf.setNOctaveLayers(value);
    }


    public Mat opticalFLow(Mat input) {
        Mat img = new Mat(), copyOfOriginal = new Mat();
        Mat flow = new Mat();
        input.copyTo(img);
        input.copyTo(copyOfOriginal);
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2GRAY);

        if (!prevgray.empty()) {
            calcOpticalFlowFarneback(prevgray, img, flow, 0.4, 1, 12, 2, 8, 1.5, 0);
            for (int y = 0; y < copyOfOriginal.rows(); y += 10)
                for (int x = 0; x < copyOfOriginal.cols(); x += 10) {
                    double flowatx = flow.get(y, x)[0] * 10;
                    double flowaty = flow.get(y, x)[1] * 10;
                    Imgproc.line(copyOfOriginal,
                            new Point(x, y),
                            new Point(Math.round(x + flowatx), Math.round(y + flowaty)),
                            new Scalar(0, 255, 0, 0));
                    Imgproc.circle(copyOfOriginal,
                            new Point(x, y),
                            2,
                            new Scalar(0, 0, 0, 0), -2, 4, 0);
                }
            img.copyTo(prevgray);
        } else {
            img.copyTo(prevgray);
        }
        return flow;
    }


    public Mat drawOpticalFlowToImage(Mat input, Mat flow) {
        Mat copyOfOriginal = new Mat();
        input.copyTo(copyOfOriginal);

        if (!flow.empty()) {
            for (int y = 0; y < copyOfOriginal.rows(); y += 10)
                for (int x = 0; x < copyOfOriginal.cols(); x += 10) {
                    double flowatx = flow.get(y, x)[0] * 10;
                    double flowaty = flow.get(y, x)[1] * 10;
                    Imgproc.line(copyOfOriginal,
                            new Point(x, y),
                            new Point(Math.round(x + flowatx), Math.round(y + flowaty)),
                            new Scalar(0, 255, 0, 0));
                    Imgproc.circle(copyOfOriginal,
                            new Point(x, y),
                            2,
                            new Scalar(0, 0, 0, 0), -2, 4, 0);
                }
        }
        return copyOfOriginal;
    }


    public Mat clusteringCoordinateDBSCAN(Mat input, MatOfKeyPoint surfKeyPoint, double eps, int minP) {
        System.out.println("Starting Clustering Position My_DBSCAN ...");
        long startTime = System.currentTimeMillis();
        Mat copyOfOriginal = new Mat();
        input.copyTo(copyOfOriginal);
        Dataset data = new DefaultDataset();
        List<KeyPoint> keyPoints = surfKeyPoint.toList();
        for (int i = 0; i < keyPoints.size(); i++) {
            KeyPoint keyPoint = keyPoints.get(i);
            double x = keyPoint.pt.x;
            double y = keyPoint.pt.y;
            //Create instance with 2 Attributes
            Instance instance = new SparseInstance(2);
            instance.put(1, x);
            instance.put(2, y);
            data.add(instance);
        }
        System.out.println("Done1");

        //===== OPTIC =====
        OPTICS optics = new OPTICS(eps, minP);
        Dataset[] cluster = optics.cluster(data);

        //===== My_DBSCAN =====
        //DensityBasedSpatialClustering dbscan = new DensityBasedSpatialClustering(eps, minP);
        //Dataset[] cluster = dbscan.cluster(data);

        System.out.println("Done2");
        for (int i = 0; i < cluster.length; i++) {
            for (int index = 0; index < cluster[i].size(); index++) {
                Instance instance = cluster[i].get(index);
                Scalar scalar;
                if (i == 0) {
                    scalar = new Scalar(0, 255, 0, 0);
                } else if (i == 1) {
                    scalar = new Scalar(0, 0, 255, 0);
                } else {
                    scalar = new Scalar(255, 0, 0, 0);
                }
                Imgproc.circle(copyOfOriginal,
                        new Point((int) instance.value(1), (int) instance.value(2)),
                        5,
                        scalar, -5, 4, 0);

            }
        }
        System.out.println("Done3");
        System.out.println("Clustering Time:" + (System.currentTimeMillis() - startTime));
        return copyOfOriginal;
    }

    private Mat clusteringTexture(Mat input, MatOfKeyPoint surfKeyPoint) {
        System.out.println("Starting Clustering Texture...");
        long startTime = System.currentTimeMillis();
        Mat labels = new Mat();
        Mat probs = new Mat();
        Mat copyOfOriginal = new Mat();
        input.copyTo(copyOfOriginal);
        List<KeyPoint> keyPoints = surfKeyPoint.toList();

        Mat samples = new Mat(new Size(3, keyPoints.size()), CvType.CV_8UC1);
        for (int i = 0; i < keyPoints.size(); i++) {
            KeyPoint keyPoint = keyPoints.get(i);
            double[] pixel = copyOfOriginal.get((int) keyPoint.pt.y, (int) keyPoint.pt.x);
            double b = pixel[0];
            double g = pixel[1];
            double r = pixel[2];
            samples.put(i, 0, b);
            samples.put(i, 1, g);
            samples.put(i, 2, r);
        }
        System.out.println("Done1");

        EM em = EM.create();
        em.setClustersNumber(3);
        samples.convertTo(samples, CvType.CV_64FC1, 1.0 / 255.0, 0.0);
        em.trainEM(samples, new Mat(), labels, probs);
        System.out.println("Done2");

        for (int i = 0; i < keyPoints.size(); i++) {
            KeyPoint keyPoint = keyPoints.get(i);
            Scalar scalar;
            if (labels.get(i, 0)[0] == 0) {
                scalar = new Scalar(255, 0, 0, 0);
            } else if (labels.get(i, 0)[0] == 1) {
                scalar = new Scalar(0, 255, 0, 0);
            } else {
                scalar = new Scalar(0, 0, 255, 0);
            }
            Imgproc.circle(copyOfOriginal,
                    new Point((int) keyPoint.pt.x, (int) keyPoint.pt.y),
                    5,
                    scalar, -5, 4, 0);
        }
        System.out.println("Done3");
        System.out.println("Clustering Time:" + (System.currentTimeMillis() - startTime));
        return copyOfOriginal;
    }

    public Mat clusteringCoordinateKmeans(MatOfKeyPoint surfKeyPoint, Mat input,
                                          Mat flow) {
        System.out.println("Starting Clustering Position Kmeans ...");
        long startTime = System.currentTimeMillis();
        List<KeyPoint> keyPoints = surfKeyPoint.toList();
        Mat labels = new Mat(new Size(1, keyPoints.size()), CvType.CV_64F);
        Mat copyOfOriginal = new Mat();
        input.copyTo(copyOfOriginal);
        //Create samples based on optical flow
        Mat samples = new Mat(new Size(2, keyPoints.size()), CvType.CV_64F);
        for (int i = 0; i < keyPoints.size(); i++) {
            KeyPoint keyPoint = keyPoints.get(i);
            double x = keyPoint.pt.x;
            double y = keyPoint.pt.y;
            samples.put(i, 0, x);
            samples.put(i, 1, y);
            if (Math.abs(flow.get((int) y, (int) x)[0]) > 1.0
                    || Math.abs(flow.get((int) y, (int) x)[1]) > 1.0) {
                labels.put(i, 0, 0);
            } else {
                labels.put(i, 0, 1);
            }
        }
        System.out.println("Done1");
        Mat centers = new Mat();
        TermCriteria criteria = new TermCriteria(
                TermCriteria.COUNT, 100, 1);
        Core.kmeans(samples, 2, labels, criteria, 3, Core.KMEANS_USE_INITIAL_LABELS, centers);
        System.out.println("Done2");

        for (int i = 0; i < keyPoints.size(); i++) {
            KeyPoint keyPoint = keyPoints.get(i);
            Scalar scalar;
            if (labels.get(i, 0)[0] == 0) {
                scalar = new Scalar(255, 0, 0, 0);
            } else if (labels.get(i, 0)[0] == 1) {
                scalar = new Scalar(0, 255, 0, 0);
            } else {
                scalar = new Scalar(0, 0, 255, 0);
            }
            Imgproc.circle(copyOfOriginal,
                    new Point((int) keyPoint.pt.x, (int) keyPoint.pt.y),
                    5,
                    scalar, -5, 4, 0);

        }
        System.out.println("Done3");
        System.out.println("Clustering Time:" + (System.currentTimeMillis() - startTime));
        return copyOfOriginal;
    }

    public Mat clusteringCoordinateGMM(Mat input, MatOfKeyPoint surfKeyPoint) {
        System.out.println("Starting Clustering Position GMM...");
        long startTime = System.currentTimeMillis();
        List<KeyPoint> keyPoints = surfKeyPoint.toList();
        Mat labels = new Mat();
        Mat probs = new Mat();
        Mat copyOfOriginal = new Mat();
        input.copyTo(copyOfOriginal);
        float width = copyOfOriginal.width();
        float height = copyOfOriginal.height();
        Mat samples = new Mat(new Size(2, keyPoints.size()), CvType.CV_64FC1);
        for (int i = 0; i < keyPoints.size(); i++) {
            KeyPoint keyPoint = keyPoints.get(i);
            double x = keyPoint.pt.x;
            double y = keyPoint.pt.y;
            samples.put(i, 0, (x / width));
            samples.put(i, 1, (y / height));
        }
        System.out.println("Done1");
        EM em = EM.create();
        em.setClustersNumber(3);
        em.setTermCriteria(new TermCriteria(TermCriteria.COUNT, 100, 1));
        em.trainEM(samples, new Mat(), labels, probs);
        System.out.println("Done2");

        for (int i = 0; i < keyPoints.size(); i++) {
            KeyPoint keyPoint = keyPoints.get(i);
            Scalar scalar;
            if (labels.get(i, 0)[0] == 0) {
                scalar = new Scalar(255, 0, 0, 0);
            } else if (labels.get(i, 0)[0] == 1) {
                scalar = new Scalar(0, 255, 0, 0);
            } else {
                scalar = new Scalar(0, 0, 255, 0);
            }
            Imgproc.circle(copyOfOriginal,
                    new Point((int) keyPoint.pt.x, (int) keyPoint.pt.y),
                    5,
                    scalar, -5, 4, 0);
        }
        System.out.println("Done3");
        System.out.println("Clustering Time:" + (System.currentTimeMillis() - startTime));
        return copyOfOriginal;
    }

    public Mat grabCut(Mat input, MatOfKeyPoint surfKeyPoint, Mat flow) {
        long start = System.currentTimeMillis();
        log("Start Grabcut ... ");
        List<KeyPoint> keyPoints = surfKeyPoint.toList();

        Mat mask = new Mat(input.size(), CvType.CV_8UC1, Scalar.all(Imgproc.GC_PR_BGD));
        Mat bgModel = new Mat(new Size(65, 1), CvType.CV_64FC1, Scalar.all(0));
        Mat fgModel = new Mat(new Size(65, 1), CvType.CV_64FC1, Scalar.all(0));
        Mat copyOfOriginal = new Mat();
        input.copyTo(copyOfOriginal);
        for (int i = 0; i < keyPoints.size(); i++) {
            Point pt = keyPoints.get(i).pt;
            mask.put((int) pt.y, (int) pt.x, (byte) Imgproc.GC_PR_FGD);
        }
        Imgproc.grabCut(input, mask,
                new Rect(20, 20, input.cols() - 20, input.rows() - 20),
                bgModel, fgModel, 5,
                Imgproc.GC_INIT_WITH_MASK);
        log("Stop Grabcut ... ");
        log("Time :" + (System.currentTimeMillis() - start));
        return mergeImageAndMask(copyOfOriginal, mask);
    }

    public Mat[] proposedModel(Mat input, MatOfKeyPoint surfKeyPoint, Mat flow) {
        List<KeyPoint> keyPoints = surfKeyPoint.toList();
        if (!initBackgroundModel) {
            backGroundModel = new Mat(input.size(), CvType.CV_8UC1, Scalar.all(Imgproc.GC_PR_BGD));
            initBackgroundModel = true;
            return new Mat[]{input};
        }
        long start = System.currentTimeMillis();
        //update background model
        log("Update background model ... ");
        for (int y = 0; y < flow.rows(); y++)
            for (int x = 0; x < flow.cols(); x++) {
                double[] ptr = backGroundModel.get(y, x);
                if (ptr[0] == (byte) Imgproc.GC_FGD) {
                    backGroundModel.put(y, x, (byte) Imgproc.GC_PR_FGD);
                }
            }

        for (int i = 0; i < keyPoints.size(); i++) {
            Point pt = keyPoints.get(i).pt;
            double floatAtX = flow.get((int) pt.y, (int) pt.x)[0];
            double floatAtY = flow.get((int) pt.y, (int) pt.x)[1];
            double[] ptr = backGroundModel.get((int) pt.y, (int) pt.x);

            if (Math.abs(floatAtX) > 1.0f && Math.abs(floatAtY) > 1.0f) {
                if (ptr[0] == (byte) Imgproc.GC_PR_FGD) {
                    backGroundModel.put((int) pt.y, (int) pt.x, (byte) Imgproc.GC_FGD);
                } else if (ptr[0] == (byte) Imgproc.GC_PR_BGD) {
                    backGroundModel.put((int) pt.y, (int) pt.x, (byte) Imgproc.GC_PR_FGD);
                }
            } else {
                if (ptr[0] == (byte) Imgproc.GC_FGD) {
                    backGroundModel.put((int) pt.y, (int) pt.x, (byte) Imgproc.GC_PR_FGD);
                } else if (ptr[0] == (byte) Imgproc.GC_PR_FGD) {
                    backGroundModel.put((int) pt.y, (int) pt.x, (byte) Imgproc.GC_PR_BGD);
                }
            }
        }

        log("Start Grabcut ... ");
        Mat mask = new Mat();
        Mat copyOfOriginal = new Mat();
        input.copyTo(copyOfOriginal);
        backGroundModel.copyTo(mask);
        mask = grabCutWithMask(input, mask);
        log("Stop Grabcut ... ");
        log("Time :" + (System.currentTimeMillis() - start));
        return new Mat[]{mergeImageAndMask(copyOfOriginal, mask)};
    }

    public Mat[] tobiModel_Upgrade(Mat input, MatOfKeyPoint surfKeyPoints, Mat flow, double eps, int minP) {
        frameCounter++;
        //Imgproc.cvtColor(input, input, Imgproc.COLOR_BGR2GRAY);
        log("Matching previous Frame to Current frame... ");
        long start = System.currentTimeMillis();
        //init - set all key points to class 0
        if (!initBackgroundModel) {
            initBackgroundModel(input, surfKeyPoints);
            return new Mat[]{input, input, input};
        }

        //use key point detector to get background point
        log("Matching previous Frame to Current frame... ");
        Mat currentFrameDescriptors = new Mat();
        surf.compute(input, surfKeyPoints, currentFrameDescriptors);
        KeyPointsAndFeaturesVector currentFrameDesScriptorsWithPosition = new KeyPointsAndFeaturesVector(surfKeyPoints, currentFrameDescriptors);
        List<MatOfDMatch> allMatchesDescriptors = matchingFeatures(currentFrameDesScriptorsWithPosition.getDescriptors(), backgroundModelTobi.getDescriptors());

        //quick calculation of max and min distances between key points
        ArrayList<DMatch> bestMatches = findBestMatches(allMatchesDescriptors);
        MatOfDMatch matOfAllMatches = new MatOfDMatch();
        matOfAllMatches.fromList(bestMatches);
        bestMatches = calculateGoodMatches(matOfAllMatches);
        Mat img_matches = drawBestMatching(input, surfKeyPoints, bestMatches);


        log("Frame counter: " + frameCounter);
        //set class of all surf point in current frame to 0
        List<KeyPoint> keyPoints = setClassToZero(surfKeyPoints);
        surfKeyPoints.fromList(keyPoints);

        List<KeyPoint> keyPointsListBackground = backgroundModelTobi.getMatOfKeyPoint().toList();
        Mat maskOnBackgroundModel = markClassOnBackgroundModel(input, keyPointsListBackground);
        ArrayList<Integer> good_indexes = updateBackgroundModelAndReturnsGoodIndexes(maskOnBackgroundModel, bestMatches, keyPoints, currentFrameDescriptors, flow);


        log("Create image with background model for testing ");

        Mat mask = createMaskFromBackgroundModel(input);

        addUnmatchedSurfToBackgroundModel(currentFrameDescriptors, keyPoints, good_indexes);

        //buildMaskImageForTesting(maskOnBackgroundModel, mask);

        log("Clustering...");
        maskOnBackgroundModel = clusteringAndDraw(surfKeyPoints, eps, minP, maskOnBackgroundModel, mask);
        log("Start grabcutting...");
        //mask = grabCutWithMask(input, mask);

        //update previous values
        input.copyTo(previousFrame);
        surfKeyPoints.copyTo(previousKeyPoint);
        currentFrameDescriptors.copyTo(previousFrameDescriptors);
        log("Finished Tobi");
        log("Time: " + (System.currentTimeMillis() - start));
        return new Mat[]{drawMaskPointToImage(input, mask), img_matches, maskOnBackgroundModel};
    }

    public Mat[] personDetector(Mat input) {
        return personDetectorAndTracking.detect4(input);
    }


    private Mat convertOFMat2BinaryMat(Mat flow) {
        Mat result = new Mat(flow.size(), CvType.CV_8UC1);
        for (int y = 0; y < flow.rows(); y++)
            for (int x = 0; x < flow.cols(); x++) {
                double[] flowAt = flow.get(y, x);
                if (Math.abs(flowAt[0]) > 0.5 || Math.abs(flowAt[1]) > 0.5) {
                    result.put(y, x, 255);
                } else {
                    result.put(y, x, 0);
                }
            }
        return result;
    }


    private Mat clusteringAndDraw(MatOfKeyPoint surfKeyPoints, double eps, int minP, Mat maskOnBackgroundModel, Mat mask) {
        Dataset data = Clustering.generateDatasetFrom_X_Y_Time(surfKeyPoints, mask);
        Clustering.My_OPTICS myOptics = new Clustering.My_OPTICS(eps, minP);
        //Dataset[] cluster = myOptics.cluster(data);
        Dataset[] cluster = new Dataset[0];
        Scalar[] colors = new Scalar[]{
                new Scalar(255, 0, 0, 0),
                new Scalar(0, 255, 0, 0),
                new Scalar(0, 0, 255, 0),
                new Scalar(255, 255, 0, 0),
                new Scalar(0, 255, 255, 0)};
        maskOnBackgroundModel = Clustering.drawClusters(maskOnBackgroundModel, cluster, colors);
        return maskOnBackgroundModel;
    }

    private void addUnmatchedSurfToBackgroundModel(Mat currentFrameDescriptors, List<KeyPoint> keyPoints, ArrayList<Integer> good_indexes) {
        for (int i = 0; i < keyPoints.size(); i++) {
            // if the surf point is NOT in background model then add them into
            if (!good_indexes.contains(i)) {
                try {
                    backgroundModelTobi.addNewKeyPointAndDescriptors(keyPoints.get(i), currentFrameDescriptors.row(i));
                } catch (KeyPointsAndFeaturesVector.KeyPointsAndFeaturesVectorException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private Mat createMaskFromBackgroundModel(Mat input) {
        Mat mask = new Mat(input.size(), CvType.CV_8UC1, Scalar.all((byte) Imgproc.GC_PR_BGD));
        List<KeyPoint> backgroundKeyPoints = backgroundModelTobi.getMatOfKeyPoint().toList();
        for (int i = 0; i < backgroundKeyPoints.size(); i++) {
            KeyPoint keyPoint = backgroundKeyPoints.get(i);
            if (keyPoint.class_id > 0) {
                mask.put((int) keyPoint.pt.y, (int) keyPoint.pt.x, Imgproc.GC_PR_FGD);
            } else {
                mask.put((int) keyPoint.pt.y, (int) keyPoint.pt.x, Imgproc.GC_PR_BGD);
            }
        }
        return mask;
    }

    private Mat markClassOnBackgroundModel(Mat input, List<KeyPoint> keyPointsListBackground) {
        Mat copyOfOriginal = new Mat();
        input.copyTo(copyOfOriginal);
        for (int i = 0; i < keyPointsListBackground.size(); i++) {
            if (keyPointsListBackground.get(i).class_id == 0) {
                Imgproc.circle(copyOfOriginal,
                        new Point((int) keyPointsListBackground.get(i).pt.x, (int) keyPointsListBackground.get(i).pt.y),
                        5,
                        new Scalar(0, 0, 0), -5, 4, 0);
            }
        }

        for (int i = 0; i < keyPointsListBackground.size(); i++) {
            if (keyPointsListBackground.get(i).class_id > 0) {
                Imgproc.circle(copyOfOriginal,
                        new Point((int) keyPointsListBackground.get(i).pt.x, (int) keyPointsListBackground.get(i).pt.y),
                        5,
                        new Scalar(255, 255, 255), -5, 4, 0);
            }
        }
        return copyOfOriginal;
    }

    private ArrayList<Integer> updateBackgroundModelAndReturnsGoodIndexes(Mat image, ArrayList<DMatch> bestMatches, List<KeyPoint> keyPoints, Mat descriptors, Mat flow) {
        log("Update background keypoints list... ");
        //compare current surf and background model
        ArrayList<Integer> good_indexes = new ArrayList<>();
        for (int i = 0; i < bestMatches.size(); i++) {
            DMatch dMatch = bestMatches.get(i);
            good_indexes.add(dMatch.queryIdx);
            KeyPoint keyPointQuery = keyPoints.get(dMatch.queryIdx);
            KeyPoint keyPointTrain = backgroundModelTobi.getKeypoint(dMatch.trainIdx);
            //KeyPoint keyPointTrain = tictacKeyPoint.toList().get(dMatch.trainIdx);

            //if key point is in background list and its position changed then update his class
            int current_class_id = keyPointTrain.class_id;
            double[] flowAt = flow.get((int) keyPointQuery.pt.y, (int) keyPointQuery.pt.x);
            //if (euclideandistance(keyPointQuery, keyPointTrain) > 10.0) {
            if (Math.abs(flowAt[0]) > 0.1 || Math.abs(flowAt[1]) > 0.1) {
                current_class_id += 1;
            }

            Imgproc.line(image,
                    new Point(keyPointTrain.pt.x, keyPointTrain.pt.y),
                    new Point(keyPointQuery.pt.x, keyPointQuery.pt.y),
                    new Scalar(0, 255, 0, 0));
            //update new position for background model
            backgroundModelTobi.getMatOfKeyPoint().put(dMatch.trainIdx, 0,
                    keyPointQuery.pt.x, keyPointQuery.pt.y,
                    keyPointQuery.size, keyPointQuery.angle, keyPointQuery.response, keyPointQuery.octave,
                    current_class_id);

            for (int col = 0; col < descriptors.cols(); col++) {
                backgroundModelTobi.getDescriptors().put(dMatch.trainIdx, col, descriptors.get(dMatch.queryIdx, col));
                backgroundModelTobi.getDescriptorsWithPosition().put(dMatch.trainIdx, col, descriptors.get(dMatch.queryIdx, col));
            }
            backgroundModelTobi.getDescriptorsWithPosition().put(dMatch.trainIdx, descriptors.cols(), keyPointQuery.pt.x / KeyPointsAndFeaturesVector.quote);
            backgroundModelTobi.getDescriptorsWithPosition().put(dMatch.trainIdx, descriptors.cols() + 1, keyPointQuery.pt.y / KeyPointsAndFeaturesVector.quote);
        }

        return good_indexes;
    }

    private List<KeyPoint> setClassToZero(MatOfKeyPoint surfKeyPoints) {
        List<KeyPoint> keyPoints = surfKeyPoints.toList();
        for (int i = 0; i < keyPoints.size(); i++) {
            KeyPoint keyPoint = keyPoints.get(i);
            keyPoint.class_id = 0;
        }
        surfKeyPoints.fromList(keyPoints);
        return keyPoints;
    }

    private List<MatOfDMatch> matchingFeatures(Mat currentFrameDescriptors, Mat backGroundModelDescriptors) {
        FlannBasedMatcher flannBasedMatcher = FlannBasedMatcher.create();
        List<MatOfDMatch> allMatches = new ArrayList<>();
        int k_neighbor = 2;
        flannBasedMatcher.knnMatch(currentFrameDescriptors, backGroundModelDescriptors, allMatches, k_neighbor);
        return allMatches;
    }

    private ArrayList<DMatch> findBestMatches(List<MatOfDMatch> allMatches) {
        ArrayList<DMatch> bestMatches = new ArrayList<>();
        for (int i = 0; i < allMatches.size(); i++) {
            List<DMatch> list = allMatches.get(i).toList();
            if (list.get(0).distance < 0.8 * list.get(1).distance) {
                bestMatches.add(list.get(0));
            }
        }
        return bestMatches;
    }

    private Mat drawBestMatching(Mat input, MatOfKeyPoint surfKeyPoints, ArrayList<DMatch> bestMatches) {
        //draw best matches to test
        Mat img_matches = new Mat();
        MatOfDMatch best_matches_Mat = new MatOfDMatch();
        best_matches_Mat.fromList(bestMatches);
        log("Best matches : " + bestMatches.size());
        Features2d.drawMatches(input, surfKeyPoints, previousFrame, backgroundModelTobi.getMatOfKeyPoint(), best_matches_Mat, img_matches);
        return img_matches;
    }

    private void initBackgroundModel(Mat input, MatOfKeyPoint surfKeyPoints) {
        input.copyTo(previousFrame);
        surf.compute(previousFrame, surfKeyPoints, previousFrameDescriptors);
        surfKeyPoints.copyTo(previousKeyPoint);
        List<KeyPoint> keyPoints = previousKeyPoint.toList();
        for (int i = 0; i < keyPoints.size(); i++) {
            KeyPoint keyPoint = keyPoints.get(i);
            keyPoint.class_id = 0;
        }
        MatOfKeyPoint newKeyPoints = new MatOfKeyPoint();
        newKeyPoints.fromList(keyPoints);
        backgroundModelTobi = new KeyPointsAndFeaturesVector(newKeyPoints, previousFrameDescriptors);
        initBackgroundModel = true;
    }

    private double euclideandistance(KeyPoint keyPoint1, KeyPoint keyPoint2) {
        return Math.sqrt(Math.pow(keyPoint1.pt.x - keyPoint2.pt.x, 2) + Math.pow(keyPoint1.pt.y - keyPoint2.pt.y, 2));
    }

    private Mat mergeImageAndMask(Mat image, Mat mask) {
        Mat newImg = new Mat();
        image.copyTo(newImg);
        for (int y = 0; y < newImg.rows(); y++)
            for (int x = 0; x < newImg.cols(); x++) {
                double maskLabel = mask.get(y, x)[0];
                if (maskLabel == 2 || maskLabel == 0) {
                    newImg.put(y, x, new byte[]{0, 0, 0});
                }
            }
        return newImg;
    }

    private Mat drawMaskPointToImage(Mat image, Mat mask) {
        Mat newImg = new Mat();
        image.copyTo(newImg);
        for (int y = 0; y < newImg.rows(); y++)
            for (int x = 0; x < newImg.cols(); x++) {
                double maskLabel = mask.get(y, x)[0];
                if (maskLabel == 1 || maskLabel == 3) {
                    Imgproc.circle(newImg,
                            new Point(x, y),
                            5,
                            new Scalar(255, 255, 255), -5, 4, 0);
                }
            }
        return newImg;
    }

    private Mat grabCutWithMask(Mat input, Mat mask) {
        Mat bgModel = new Mat(new Size(65, 1), CvType.CV_64FC1, Scalar.all(0));
        Mat fgModel = new Mat(new Size(65, 1), CvType.CV_64FC1, Scalar.all(0));
        Imgproc.grabCut(input, mask,
                new Rect(20, 20, input.cols() - 20, input.rows() - 20),
                bgModel, fgModel, 5,
                Imgproc.GC_INIT_WITH_MASK);
        return mask;
    }


    private ArrayList<DMatch> calculateGoodMatches(MatOfDMatch matches) {
        double max_dist = 0;
        double min_dist = 100;
        //-- Quick calculation of max and min distances between key points
        List<DMatch> dMatches = matches.toList();
        for (int i = 0; i < dMatches.size(); i++) {
            double dist = dMatches.get(i).distance;
            if (dist < min_dist) min_dist = dist;
            if (dist > max_dist) max_dist = dist;
        }

        ArrayList<DMatch> goodMatches = new ArrayList<>();
        for (int i = 0; i < dMatches.size(); i++) {
            if (dMatches.get(i).distance <= Math.max(2 * min_dist, 0.02)) {
                goodMatches.add(dMatches.get(i));
            }
        }
        return goodMatches;
    }

    private void log(Object o) {
        System.out.println(o);
    }
}
