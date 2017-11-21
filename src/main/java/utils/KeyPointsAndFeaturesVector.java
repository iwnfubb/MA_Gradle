package utils;


import imageprocess.ImageProcess_ObjectTracking;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.List;

public class KeyPointsAndFeaturesVector {
    private MatOfKeyPoint matOfKeyPoint;
    private Mat descriptors;
    private Mat descriptorsWithPosition;

    public KeyPointsAndFeaturesVector() {
        this.matOfKeyPoint = new MatOfKeyPoint();
        this.descriptors = new Mat();
        this.descriptorsWithPosition = new Mat();
    }

    public KeyPointsAndFeaturesVector(MatOfKeyPoint keypointVector, Mat descriptors) {
        this.matOfKeyPoint = keypointVector;
        this.descriptors = descriptors;
        descriptorsWithPosition = new Mat(0, descriptors.cols() + 2, descriptors.type());
        for (int i = 0; i < descriptors.rows(); i++) {
            Mat row = descriptors.row(i);
            Imgproc.resize(row, row, new Size(descriptors.cols() + 2, 1));
            double x = matOfKeyPoint.get(i, 0)[0];
            double y = matOfKeyPoint.get(i, 0)[1];
            row.put(0, descriptors.cols(), x);
            row.put(0, descriptors.cols() + 1, y);
            descriptorsWithPosition.push_back(row);
        }
    }

    public void addNewKeyPointAndDescriptors(KeyPoint keyPoint, Mat descriptor) throws KeyPointsAndFeaturesVectorException {
        if (descriptor.cols() == descriptors.cols()) {
            Mat keypointMat = new Mat(1, 1, CvType.CV_32FC(7));
            keypointMat.put(0, 0, keyPoint.pt.x, keyPoint.pt.y, keyPoint.size, keyPoint.angle, keyPoint.response, keyPoint.octave, keyPoint.class_id);
            Mat desWithPos = new Mat();
            descriptor.copyTo(desWithPos);
            Imgproc.resize(desWithPos, desWithPos, new Size(descriptors.cols() + 2, 1));
            desWithPos.put(0, descriptors.cols(), keyPoint.pt.x);
            desWithPos.put(0, descriptors.cols() + 1, keyPoint.pt.y);
            matOfKeyPoint.push_back(keypointMat);
            descriptors.push_back(descriptor);
            descriptorsWithPosition.push_back(desWithPos);
        } else {
            throw new KeyPointsAndFeaturesVectorException("new col:" + descriptor.cols() + " is not equal " + +descriptors.cols());
        }
    }

    public KeyPoint getKeypoint(int index) {
        return matOfKeyPoint.toList().get(index);
    }

    public Mat getDescriptor(int index) {
        return descriptors.row(index);
    }

    public class KeyPointsAndFeaturesVectorException extends Exception {
        public KeyPointsAndFeaturesVectorException(String msg) {
            super(msg);
        }
    }

    public MatOfKeyPoint getMatOfKeyPoint() {
        return matOfKeyPoint;
    }

    public void setMatOfKeyPoint(MatOfKeyPoint matOfKeyPoint) {
        this.matOfKeyPoint = matOfKeyPoint;
    }

    public Mat getDescriptors() {
        return descriptors;
    }

    public void setDescriptors(Mat descriptors) {
        this.descriptors = descriptors;
    }

    public Mat getDescriptorsWithPosition() {
        return descriptorsWithPosition;
    }

    public void setDescriptorsWithPosition(Mat descriptorsWithPosition) {
        this.descriptorsWithPosition = descriptorsWithPosition;
    }

    public Mat getDescriptorsWithPosition(int index) {
        return descriptorsWithPosition.row(index);
    }
}
