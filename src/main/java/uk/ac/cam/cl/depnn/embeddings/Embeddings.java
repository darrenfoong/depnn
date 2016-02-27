package uk.ac.cam.cl.depnn.embeddings;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Embeddings {
	private final static int UNK = 0;
	private final static String UNK_STRING = "_UNK_";

	private HashMap<String, Integer> map;
	private double[][] embeddings;

	public Embeddings(HashSet<String> lexicon, int sizeEmbeddings, double randomRange) {
		int numEmbeddings = lexicon.size() + 1;

		embeddings = new double[numEmbeddings][sizeEmbeddings];

		// starts from 1 because of UNK
		Iterator<String> iter = lexicon.iterator();

		map.put(UNK_STRING, 0);

		for ( int count = 1; count < numEmbeddings; count++ ) {
			map.put(iter.next(), count);
		}

		randomWeights(randomRange);
	}

	public Embeddings(String embeddingsFile) throws IOException {
		int numEmbeddings = 0;
		int sizeEmbeddings = 0;
		int count = 0;

		try ( BufferedReader in = new BufferedReader(new FileReader(embeddingsFile)) ) {
			// first pass to count numEmbeddings and sizeEmbeddings
			String line = in.readLine();
			sizeEmbeddings = line.split(" ").length - 1;
			numEmbeddings++;

			while ( in.readLine() != null ) {
				numEmbeddings++;
			}

			embeddings = new double[numEmbeddings][sizeEmbeddings];

			in.reset();

			// second pass to read embeddings
			while ( (line = in.readLine()) != null ) {
				String[] lineSplit = line.split(" ");
				map.put(lineSplit[0], count);

				for ( int i = 0; i < sizeEmbeddings; i++ ) {
					embeddings[count][0] = Double.parseDouble(lineSplit[1+i]);
				}

				count++;
			}
		} catch ( IOException e ) {
			throw e;
		}
	}

	public void serializeEmbeddings(String embeddingsFile) throws IOException {
		try ( PrintWriter out = new PrintWriter(new FileWriter(embeddingsFile)) ) {
			String[] keys = new String[embeddings.length];

			// get mapping from values to keys
			for ( Map.Entry<String, Integer> entry : map.entrySet() ) {
				keys[entry.getValue()] = entry.getKey();
			}

			for ( int i = 0; i < embeddings.length; i++ ) {
				StringBuilder outBuilder = new StringBuilder(keys[i]);

				for ( int j = 0; j < embeddings[i].length; j++ ) {
					outBuilder.append(embeddings[i][j]);
					outBuilder.append(" ");
				}

				out.println(outBuilder.toString());
			}
		} catch ( IOException e ) {
			throw e;
		}
	}

	private void randomWeights(double randomRange) {
		for ( int i = 0; i < embeddings.length; i++ ) {
			for ( int j = 0; j < embeddings[i].length; j++ ) {
				embeddings[i][j] = Math.random() * 2 * randomRange - randomRange;
			}
		}
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
