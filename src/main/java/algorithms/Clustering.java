package algorithms;

import net.sf.javaml.clustering.DensityBasedSpatialClustering;
import net.sf.javaml.clustering.OPTICS;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.core.SparseInstance;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import utils.Parameters;

import java.util.List;

public class Clustering {
    private double eps = 0.05;
    private int minP = 10;

    public Clustering(double eps, int minP) {
        this.eps = eps;
        this.minP = minP;
    }

    public double getEps() {
        return eps;
    }

    public void setEps(double eps) {
        this.eps = eps;
    }

    public int getMinP() {
        return minP;
    }

    public void setMinP(int minP) {
        this.minP = minP;
    }


    public static Mat drawClusters(Mat input, Dataset[] cluster, Scalar... colors) {
        Mat copy_of_input = new Mat();
        input.copyTo(copy_of_input);
        for (int i = 0; i < cluster.length; i++) {
            for (int index = 0; index < cluster[i].size(); index++) {
                Instance instance = cluster[i].get(index);
                Scalar scalar = Parameters.color_white;
                if (i < colors.length) {
                    scalar = colors[i];
                }
                Imgproc.circle(copy_of_input,
                        new Point((int) instance.value(1), (int) instance.value(2)),
                        5,
                        scalar, -5, 4, 0);
            }
        }
        return copy_of_input;
    }

    public static Dataset generateDatasetFrom_X_Y_Time(MatOfKeyPoint keypointsMat, Mat time) {
        Dataset data = new DefaultDataset();
        List<KeyPoint> keyPoints = keypointsMat.toList();
        for (int i = 0; i < keyPoints.size(); i++) {
            KeyPoint keyPoint = keyPoints.get(i);
            //Create instance with 3 Attributes (x, y, class)
            Instance instance = new SparseInstance(3);
            instance.put(1, keyPoint.pt.x);
            instance.put(2, keyPoint.pt.y);
            instance.put(3, Double.parseDouble(String.valueOf(time.get((int) keyPoint.pt.y, (int) keyPoint.pt.x)[0])));
            data.add(instance);
        }
        return data;
    }


    public static class My_DBSCAN extends Clustering {
        DensityBasedSpatialClustering dbscan;

        public My_DBSCAN(double eps, int minP) {
            super(eps, minP);
            this.dbscan = new DensityBasedSpatialClustering(eps, minP);
        }

        public Dataset[] cluster(Dataset dataset) {
            return dbscan.cluster(dataset);
        }
    }

    public static class My_OPTICS extends Clustering {
        OPTICS myOptics;

        public My_OPTICS(double eps, int minP) {
            super(eps, minP);
            this.myOptics = new OPTICS(eps, minP);
        }

        public Dataset[] cluster(Dataset dataset) {
            return myOptics.cluster(dataset);
        }
    }

}
