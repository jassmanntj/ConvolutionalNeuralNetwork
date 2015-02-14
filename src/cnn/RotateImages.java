package cnn;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.Highgui;

import java.io.File;
import java.io.IOException;

/**
 * Created by jassmanntj on 2/11/2015.
 */
public class RotateImages {

    public static void main(String[] args) throws IOException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        rotateImgs();
    }

    public static void rotateImgs() throws IOException {
        int z = 0;
        double degrees = 5;
        File folder = new File("C:/Users/jassmanntj/Desktop/TrainSort");
        for(File leaf : folder.listFiles()) {
            if(leaf.isDirectory()) {
                System.out.println(z++ +":"+leaf.getName());
                for(File photos : leaf.listFiles()) {
                    File newdir = new File(photos.getAbsolutePath().replace("TrainSort", "TrainRot"+degrees));
                    if(!newdir.exists())
                        newdir.mkdirs();
                    for(File image : photos.listFiles()) {
                        Mat src = Highgui.imread(image.getAbsolutePath());
                        Point center = new Point(src.cols() / 2.0, src.rows() / 2.0);
                        Rect bb = new RotatedRect(center, src.size(), degrees).boundingRect();
                        Mat rot = Imgproc.getRotationMatrix2D(center, degrees, 1);
                        rot.put(0,2,rot.get(0,2)[0]+bb.width/2.0-center.x);
                        rot.put(1,2,rot.get(1,2)[0]+bb.height/2.0-center.y);
                        Mat dst = new Mat();
                        Imgproc.warpAffine(src, dst, rot, bb.size());
                        String path = image.getAbsolutePath().replace("TrainSort","TrainRot"+degrees);
                        Highgui.imwrite(path, dst);
                    }
                }
            }
        }

    }
}
