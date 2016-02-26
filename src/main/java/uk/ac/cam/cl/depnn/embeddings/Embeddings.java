package uk.ac.cam.cl.depnn.embeddings;

import java.util.HashMap;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Embeddings {
	private final static int UNK = 0;

	private HashMap<String, Integer> map;
	private double[][] embeddings;

	public Embeddings(String embeddingsFile) {
		int numEmbeddings = 0;
		int sizeEmbeddings = 0;

		// read embeddings
		// embedding 0 is the UNK embedding

		embeddings = new double[numEmbeddings][sizeEmbeddings];
	}

	public double[] getArray(String key) {
		if ( map.containsKey(key) ) {
			return embeddings[map.get(key)].clone();
		} else {
			return embeddings[UNK].clone();
		}
	}

	public INDArray getINDArray(String key) {
		double[] array = getArray(key);
		return null;
	}

	public void setEmbedding(String key, double[] embedding) {
		if ( map.containsKey(key) ) {
			embeddings[map.get(key)] = embedding.clone();
		}
	}

	public void setEmbedding(String key, INDArray embedding) {
		double[] array = {0.0};
		setEmbedding(key, array);
	}
}
