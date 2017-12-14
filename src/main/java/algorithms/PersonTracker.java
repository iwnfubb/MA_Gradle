package algorithms;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.tracking.Tracker;
import org.opencv.tracking.TrackerKCF;
import utils.Utils;

public class PersonTracker {
    Tracker tracker;
    private Rect2d trackingBox;
    private Rect2d last_trackingBox;

    public PersonTracker() {
        createTracker();
    }

    public void createTracker() {
        this.tracker = TrackerKCF.create();
    }

    public void createTracker(Mat input, Rect rect) {
        this.tracker = TrackerKCF.create();
        setTrackingBox(rect);
        initTrackingbox(input);
        saveTrackingBoxToMemory();
    }

    public void createTracker(Mat input, double[] rect) {
        this.tracker = TrackerKCF.create();
        setTrackingBox(rect);
        initTrackingbox(input);
    }

    public void initTrackingbox(Mat input) {
        tracker.init(input, trackingBox);
    }

    public void setTrackingBox(Rect rect) {
        trackingBox = Utils.convertRectToRect2d(rect);
    }

    public void setTrackingBox(double[] rect) {
        Rect r = Utils.convertDoubleToRect(rect);
        setTrackingBox(r);
    }

    public void saveTrackingBoxToMemory() {
        last_trackingBox = new Rect2d(trackingBox.x, trackingBox.y, trackingBox.width, trackingBox.height);
    }

    public void restoreCurrentTrackingBoxFromLast() {
        trackingBox = new Rect2d(last_trackingBox.x, last_trackingBox.y, last_trackingBox.width, last_trackingBox.height);
    }

    public boolean isNormalChange() {
        Rect last_rect = Utils.convertRect2dToRect(last_trackingBox);
        Rect rect = Utils.convertRect2dToRect(trackingBox);
        if (!Utils.similarArea(rect, last_rect)) {
            return false;
        }
        if (Utils.euclideandistance(last_rect, rect) > 10) {
            return false;
        }
        return true;
    }


    public Rect2d getTrackingBoxAsRect2d() {
        return trackingBox;
    }

    public Rect getTrackingBoxAsRect() {
        return Utils.convertRect2dToRect(trackingBox);
    }

    public MatOfRect getTrackingBoxAsMatOfRect() {
        return Utils.convertRect2dToMatOfRect(trackingBox);
    }


    public Rect2d getLast_TrackingBoxAsRect2d() {
        return last_trackingBox;
    }

    public Rect getLast_TrackingBoxAsRect() {
        return Utils.convertRect2dToRect(last_trackingBox);
    }

    public MatOfRect getLast_TrackingBoxAsMatOfRect() {
        return Utils.convertRect2dToMatOfRect(last_trackingBox);
    }

    public boolean updateTrackingbox(Mat input) {
        return tracker.update(input, trackingBox);
    }

}
