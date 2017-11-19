package utils;


import org.opencv.core.CvType;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;

import java.util.List;

public class KeyPointsAndFeaturesVector {
    private MatOfKeyPoint matOfKeyPoint;
    private Mat descriptors;

    public KeyPointsAndFeaturesVector() {
        this.matOfKeyPoint = new MatOfKeyPoint();
        this.descriptors = new Mat();
    }

    public KeyPointsAndFeaturesVector(MatOfKeyPoint keypointVector, Mat descriptors) {
        this.matOfKeyPoint = keypointVector;
        this.descriptors = descriptors;
    }

    public void addNewKeyPointAndDescriptors(KeyPoint keyPoint, Mat descriptor) throws KeyPointsAndFeaturesVectorException {
        if (descriptor.cols() == descriptors.cols()) {
            Mat keypointMat = new Mat(1, 1, CvType.CV_32FC(7));
            keypointMat.put(0,0, keyPoint.pt.x, keyPoint.pt.y, keyPoint.size, keyPoint.angle, keyPoint.response, keyPoint.octave, keyPoint.class_id);
            matOfKeyPoint.push_back(keypointMat);
            descriptors.push_back(descriptor);
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
}
