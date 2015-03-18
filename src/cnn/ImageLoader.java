package cnn;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.imageio.ImageIO;
import javax.xml.crypto.Data;

import org.jblas.DoubleMatrix;
import org.imgscalr.Scalr;
import org.imgscalr.Scalr.Method;
import org.imgscalr.Scalr.Rotation;

public class ImageLoader {
	DoubleMatrix[][] images;
	DoubleMatrix labels;
	int channels;
	int width, height;
	HashMap<String, Double> labelMap;
	DoubleMatrix[][] photoImages;
	DoubleMatrix photoLabels;
	ArrayList<String> photoNames;
	ArrayList<String> names;
	
	public void loadFolder(File folder, int channels, int width, int height, HashMap<String, Double> labelMap) throws IOException {
		int z = 0;
		this.channels = channels;
		this.width = width;
		this.height = height;
		this.labelMap = labelMap;
		this.photoNames = new ArrayList<String>();
		this.names = new ArrayList<String>();
        int[] counts = countImages(folder, labelMap);
        images = new DoubleMatrix[counts[0]][channels];
        photoImages = new DoubleMatrix[counts[1]][channels];
        labels = new DoubleMatrix(counts[0], labelMap.size());
        photoLabels = new DoubleMatrix(counts[1], labelMap.size());
        DoubleMatrix imageNo = new DoubleMatrix(counts[0]);
        for(int i = 0; i < imageNo.length; i++) {
            imageNo.put(i, i);
        }
        Random rand = new Random(System.currentTimeMillis());
        for(int i = 0; i < 10000; i++) {
            if(i%1000==0) System.out.println(i);
            int a = rand.nextInt(images.length);
            int b = rand.nextInt(images.length);
            imageNo.swapRows(a, b);
        }
        int i = 0;
        int j = 0;
        ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
		for(File leaf : folder.listFiles()) {
			if(leaf.isDirectory() && labelMap.containsKey(leaf.getName())) {
				//System.out.println(z++ +":"+leaf.getName());
				for(File photos : leaf.listFiles()) {
					//if(photos.getName().equals("photograph")) {
						for(File image : photos.listFiles()) {
                            boolean photo = photos.getName().equals("photograph");
                            Runnable ip = new ImageProcessor(image, i, j, labelMap.get(leaf.getName()), photo, imageNo);
                            executor.execute(ip);
                            i++;
                            if(photo) j++;
						}
					//}
				}
			}
		}
        executor.shutdown();
        while(!executor.isTerminated());
	}

    private class ImageProcessor implements Runnable {
        private File image;
        private double leafNo;
        private int num;
        private int photoNum;
        private int imgNo;
        private boolean photo;

        public ImageProcessor(File image, int num, int photoNum, double leafNo, boolean photo, DoubleMatrix imageNo) {
            this.image = image;
            this.num = (int)imageNo.get(num);
            this.imgNo = num;

            this.leafNo = leafNo;
            this.photo = photo;
            this.photoNum = photoNum;
        }

        public void run() {
            try {
                BufferedImage img = ImageIO.read(image);
                if (img.getHeight() < img.getWidth()) {
                    img = Scalr.rotate(img, Rotation.CW_90);
                }
                if (img.getHeight() * 3 < img.getWidth() * 4) {
                    BufferedImage newImage = new BufferedImage(img.getWidth(), img.getWidth() * 4 / 3, img.getType());
                    Graphics g = newImage.getGraphics();
                    g.setColor(Color.black);
                    g.fillRect(0, 0, newImage.getWidth(), newImage.getHeight());
                    g.drawImage(img, 0, (newImage.getHeight() - img.getHeight()) / 2, null);
                    g.dispose();
                    img = newImage;
                    newImage.flush();
                } else if (img.getHeight() * 3 > img.getWidth() * 4) {
                    BufferedImage newImage = new BufferedImage(img.getHeight() * 3 / 4, img.getHeight(), img.getType());
                    Graphics g = newImage.getGraphics();
                    g.setColor(Color.black);
                    g.fillRect(0, 0, newImage.getWidth(), newImage.getHeight());
                    g.drawImage(img, (newImage.getWidth() - img.getWidth()) / 2, 0, null);
                    g.dispose();
                    img = newImage;
                    newImage.flush();
                }
                img = Scalr.resize(img, Method.QUALITY, width, height);
                int[] pixels = ((DataBufferInt) img.getRaster().getDataBuffer()).getData();
                img.flush();
                if (pixels.length == width * height) {
                    for (int i = 0; i < pixels.length; i++) {
                        for (int j = 0; j < channels; j++) {
                            if(images[num][j] == null) images[num][j] = new DoubleMatrix(height,width);
                            images[num][j].put(i, ((pixels[i] >>> (8 * j)) & 0xFF));
                            if (photo) {
                                if(photoImages[photoNum][j] == null) photoImages[photoNum][j] = new DoubleMatrix(height,width);
                                photoImages[photoNum][j].put(i, ((pixels[i] >>> (8 * j)) & 0xFF));
                            }
                        }
                    }
                    labels.put(num, (int)leafNo, 1);
                    if (photo) {
                        photoLabels.put(photoNum, (int)leafNo, 1);
                    }
                }
                System.out.println(imgNo);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private int[] countImages(File folder, HashMap<String, Double> labelMap) {
        int count[] = new int[2];
        for(File leaf : folder.listFiles()) {
            if (leaf.isDirectory() && labelMap.containsKey(leaf.getName())) {
                for (File photos : leaf.listFiles()) {
                    //if(photos.getName().equals("photograph")) {
                    for (File image : photos.listFiles()) {
                        count[0]++;
                        if(photos.getName().equals("photograph")) count[1]++;
                    }
                }
            }
        }
        return count;
    }
	
	public DataContainer getPhotoImages() {
		return new DataContainer(photoImages);
	}	
	public DoubleMatrix getPhotoLabels() {
		/*DoubleMatrix expandLabels = DoubleMatrix.zeros(photoLabels.rows, labelMap.size());
		for(int i = 0; i < photoLabels.rows; i++) {
			expandLabels.put(i, (int)photoLabels.get(i),1);
		}
		return expandLabels;*/
        return photoLabels;
	}
	public ArrayList<String> getNames(File folder) {
		names = new ArrayList<String>();
		photoNames = new ArrayList<String>();
		for(File leaf : folder.listFiles()) {
			if(leaf.isDirectory()) {
				for(File photos : leaf.listFiles()) {
					for(File image : photos.listFiles()) {
						names.add(image.getName());
						if(photos.getName().equals("photograph")) {
							photoNames.add(image.getName());
						}
					}
				}
			}
        }
		return names;
	}
	public ArrayList<String> getPhotoNames() {
		return photoNames;
	}

    public static DataContainer sample(final int patchRows, final int patchCols, final int numPatches, final DataContainer im, final int width, final int height, final int channels) {
        final DoubleMatrix[][] patches = new DoubleMatrix[numPatches][channels];
        final DoubleMatrix[][] images = im.getDataArray();
        class Patcher implements Runnable {
            private int threadNo;

            public Patcher(int threadNo) {
                this.threadNo = threadNo;
            }

            @Override
            public void run() {
                int count = numPatches / Utils.NUMTHREADS;
                for (int i = 0; i < count; i++) {
                    if(i%500 == 0)
                        System.out.println(threadNo+":"+i);
                    Random rand = new Random();
                    int randomImage = rand.nextInt(images.length);
                    int randomY = rand.nextInt(height - patchRows + 1);
                    int randomX = rand.nextInt(width - patchCols + 1);
                    for (int j = 0; j < channels; j++) {
                        DoubleMatrix channel = images[randomImage][j];
                        channel = channel.getRange(randomY, randomY + patchRows, randomX, randomX + patchCols);
                        channel = channel.reshape(1, patchRows * patchCols);
                        patches[threadNo * count + i][j] = channel;
                    }
                }
            }
        }
        ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
        for(int i = 0; i < Utils.NUMTHREADS; i++) {
            if (i % 1000 == 0)
                System.out.println(i);
            Runnable patcher = new Patcher(i);
            executor.execute(patcher);
        }
        executor.shutdown();
        while(!executor.isTerminated());
        return new DataContainer(normalizeData(patches));
    }


	
	public static DoubleMatrix[][] normalizeData(DoubleMatrix[][] data) {
        double var = 0;
        for(int i = 0; i < data.length; i++) {
            double mean = 0;
            for(int j = 0; j < data[i].length; j++) {
                mean += data[i][j].mean();
            }
            for(int j = 0; j < data[i].length; j++) {
                data[i][j].subi(mean);
                var += data[i][j].mul(data[i][j]).mean()/data[i].length;
            }
        }
        var /= data.length;

		double stdev = Math.sqrt(var);
		double pstd = 3 * stdev;
		for(int i = 0; i < data.length; i++) {
			for(int j = 0; j < data[i].length; j++) {
                for(int k = 0; k < data[i][j].length; k++) {
                    double x = data[i][j].get(k);
                    double val = x < pstd ? x : pstd;
                    val = val > -pstd ? val : -pstd;
                    val /= pstd;
                    data[i][j].put(k, val);
                }
			}
		}
		//data.addi(1).muli(.4).add(0.1);
		return data;
	}
	
	public DataContainer getImages() {
		return new DataContainer(images);
	}
	
	public DoubleMatrix getLabels() {
		//DoubleMatrix expandLabels = DoubleMatrix.zeros(labels.rows, labelMap.size());
		//for(int i = 0; i < labels.rows; i++) {
		//	expandLabels.put(i, (int)labels.get(i),1);
		//}
		//return expandLabels;
        return labels;
	}
	
	public HashMap<String, Double> getLabelMap(File folder) {
		HashMap<String, Double> labelMap = new HashMap<String, Double>();
		double labelNo = -1;
		for(File leaf : folder.listFiles()) {
            if(leaf.isDirectory()) {
                labelNo++;
                leaf.listFiles();
                labelMap.put(leaf.getName(), labelNo);
            }
		}
		return labelMap;
	}
}
