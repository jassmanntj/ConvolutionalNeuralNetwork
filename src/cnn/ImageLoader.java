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

import org.jblas.DoubleMatrix;
import org.imgscalr.Scalr;
import org.imgscalr.Scalr.Method;
import org.imgscalr.Scalr.Rotation;

public class ImageLoader {
	DoubleMatrix images;
	DoubleMatrix labels;
	int channels;
	int width, height;
	HashMap<String, Double> labelMap;
	DoubleMatrix photoImages;
	DoubleMatrix photoLabels;
	ArrayList<String> photoNames;
	ArrayList<String> names;
	
	public void loadFolder(File folder, int channels, int width, int height, HashMap<String, Double> labelMap) throws IOException {
		images = null;
		photoImages = null;
		labels = null;
		photoLabels = null;
		int z = 0;
		this.channels = channels;
		this.width = width;
		this.height = height;
		this.labelMap = labelMap;
		this.photoNames = new ArrayList<String>();
		this.names = new ArrayList<String>();
		for(File leaf : folder.listFiles()) {
			if(leaf.isDirectory()) {
				System.out.println(z++ +":"+leaf.getName());
				for(File photos : leaf.listFiles()) {
					//if(photos.getName().equals("photograph")) {
						for(File image : photos.listFiles()) {
							BufferedImage img = ImageIO.read(image);
							int[] pixels = new int[1];
							if(img.getHeight() < img.getWidth()) {
								img = Scalr.rotate(img, Rotation.CW_90);
							}
							if(img.getHeight() * 3 < img.getWidth() * 4) {
								BufferedImage newImage = new BufferedImage(img.getWidth(), img.getWidth()*4/3, img.getType());
								Graphics g = newImage.getGraphics();
								g.setColor(Color.black);
								g.fillRect(0,0,newImage.getWidth(),newImage.getHeight());
								g.drawImage(img, 0, (newImage.getHeight() - img.getHeight())/2, null);
								g.dispose();
								img =  newImage;
								newImage.flush();
							}
							else if(img.getHeight() * 3 > img.getWidth() * 4) {
								BufferedImage newImage = new BufferedImage(img.getHeight() * 3/4, img.getHeight(), img.getType());
								Graphics g = newImage.getGraphics();
								g.setColor(Color.black);
								g.fillRect(0,0,newImage.getWidth(),newImage.getHeight());
								g.drawImage(img, (newImage.getWidth() - img.getWidth())/2, 0, null);
								g.dispose();
								img =  newImage;
								newImage.flush();
							}
							img = Scalr.resize(img, Method.QUALITY, width, height);
							pixels = ((DataBufferInt)img.getRaster().getDataBuffer()).getData();
							img.flush();
							if(pixels.length==width*height) {
								DoubleMatrix row = new DoubleMatrix(1, pixels.length*channels);
								DoubleMatrix lRow = new DoubleMatrix(1,1);
								lRow.put(0, labelMap.get(leaf.getName()));
								for(int i = 0; i < pixels.length; i++) {
									for(int j = 0; j < channels; j++) {
										row.put(j*pixels.length+i, ((pixels[i]>>>(8*j)) & 0xFF));
									}
								}
								if(images == null) {
									images = row;
									labels = lRow;
								}
								else {
									labels = DoubleMatrix.concatVertically(labels, lRow);
									images = DoubleMatrix.concatVertically(images, row);
								}
								names.add(image.getName());
								if(photos.getName().equals("photograph")) {
									photoNames.add(image.getName());
									if(photoImages == null) {
										photoImages = row;
										photoLabels = lRow;
									}
									else {
										photoImages = DoubleMatrix.concatVertically(photoImages, row);
										photoLabels = DoubleMatrix.concatVertically(photoLabels, lRow);
									}
								}
								
							}
						}
					//}
				}
			}
		}
	}
	
	public DoubleMatrix getPhotoImages() {
		return photoImages;
	}	
	public DoubleMatrix getPhotoLabels() {
		DoubleMatrix expandLabels = DoubleMatrix.zeros(photoLabels.rows, labelMap.size());
		for(int i = 0; i < photoLabels.rows; i++) {
			expandLabels.put(i, (int)photoLabels.get(i),1);
		}
		return expandLabels;
	}
	public ArrayList<String> getNames(File folder) {
		if(names == null) {
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
		}
		return names;
	}
	public ArrayList<String> getPhotoNames() {
		return photoNames;
	}
	
	public static DoubleMatrix sampleS(int patchSize, int numPatches, DoubleMatrix images, int width, int height, int channels) {
		Random rand = new Random();
		DoubleMatrix patches = null;
		for(int i = 0; i < numPatches; i++) {
			if(i%1000==0)
				System.out.println(i);
			int randomImage = rand.nextInt(images.rows);
			int randomY = rand.nextInt(height-patchSize+1);
			int randomX = rand.nextInt(width-patchSize+1);
			int imageSize = width*height;
			DoubleMatrix patch = null;
			for(int j = 0; j < channels; j++) {
				DoubleMatrix channel = images.getRange(randomImage, randomImage+1, j*imageSize, (j+1)*imageSize);
				channel = channel.reshape(width, height);
				channel = channel.getRange(randomY, randomY+patchSize, randomX, randomX+patchSize);
				channel = channel.reshape(1,  patchSize*patchSize);
				if(patch == null) {
					patch = channel;
				}
				else {
					patch = DoubleMatrix.concatHorizontally(patch, channel);
				}
			}
			if(patches == null) {
				patches = patch;
			}
			else {
				patches = DoubleMatrix.concatVertically(patches, patch);
			}
		}
		return normalizeData(patches);
	}

    public static DoubleMatrix sample(final int patchSize, final int numPatches, final DoubleMatrix images, final int width, final int height, final int channels) {
        DoubleMatrix patches = new DoubleMatrix(numPatches, patchSize*patchSize*channels);
        class Patcher implements Runnable {
            private int threadNo;
            private DoubleMatrix patches;

            public Patcher(int threadNo, DoubleMatrix patches) {
                this.threadNo = threadNo;
                this.patches = patches;
            }

            @Override
            public void run() {
                int count = numPatches / Utils.NUMTHREADS;
                for (int i = 0; i < count; i++) {
                    if(i%500 == 0)
                        System.out.println(threadNo+":"+i);
                    Random rand = new Random();
                    int randomImage = rand.nextInt(images.rows);
                    int randomY = rand.nextInt(height - patchSize + 1);
                    int randomX = rand.nextInt(width - patchSize + 1);
                    int imageSize = width * height;
                    DoubleMatrix patch = null;
                    for (int j = 0; j < channels; j++) {
                        DoubleMatrix channel = images.getRange(randomImage, randomImage + 1, j * imageSize, (j + 1) * imageSize);
                        channel = channel.reshape(width, height);
                        channel = channel.getRange(randomY, randomY + patchSize, randomX, randomX + patchSize);
                        channel = channel.reshape(1, patchSize * patchSize);
                        if (patch == null) {
                            patch = channel;
                        } else {
                            patch = DoubleMatrix.concatHorizontally(patch, channel);
                        }
                    }
                    patches.putRow(threadNo * count + i, patch);
                }
            }
        }
        ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
        for(int i = 0; i < Utils.NUMTHREADS; i++) {
            if (i % 1000 == 0)
                System.out.println(i);
            Runnable patcher = new Patcher(i, patches);
            executor.execute(patcher);
        }
        executor.shutdown();
        while(!executor.isTerminated());
        return normalizeData(patches);
    }


	
	public static DoubleMatrix normalizeData(DoubleMatrix data) {
		DoubleMatrix mean = data.rowMeans();
		data.subiColumnVector(mean);
		DoubleMatrix squareData = data.mul(data);
		
		double var = squareData.mean();
		double stdev = Math.sqrt(var);
		double pstd = 3 * stdev;
		for(int i = 0; i < data.rows; i++) {
			for(int j = 0; j < data.columns; j++) {
				double x = data.get(i, j);
				double val = x<pstd?x:pstd;
				val = val>-pstd?val:-pstd;
				val /= pstd;
				data.put(i, j, val);
			}
		}
		data.addi(1).muli(.4).add(0.1);
		return data;
	}
	
	public DoubleMatrix getImages() {
		return images;
	}
	
	public DoubleMatrix getLabels() {
		DoubleMatrix expandLabels = DoubleMatrix.zeros(labels.rows, labelMap.size());
		for(int i = 0; i < labels.rows; i++) {
			expandLabels.put(i, (int)labels.get(i),1);
		}
		return expandLabels;
	}
	
	public HashMap<String, Double> getLabelMap(File folder) {
		HashMap<String, Double> labelMap = new HashMap<String, Double>();
		double labelNo = -1;
		for(File leaf : folder.listFiles()) {
			labelNo++;
			leaf.listFiles();
			labelMap.put(leaf.getName(), labelNo);
		}
		return labelMap;
	}
}
