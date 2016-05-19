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
import java.util.Random;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Embeddings {
	protected int numEmbeddings = 0;
	protected int sizeEmbeddings = 0;
	protected int UNK = 0;

	private final static String UNK_STRING = "_UNK_";

	/*
	 * assumption: no values are null (for performance i.e. avoid calls to
	 * containKey())
	 */

	private final static Logger logger = LogManager.getLogger(Embeddings.class);

	protected HashMap<String, Integer> map = new HashMap<String, Integer>();
	protected double[][] embeddings;

	public int getNumEmbeddings() {
		return numEmbeddings;
	}

	public int getSizeEmbeddings() {
		return sizeEmbeddings;
	}

	public HashMap<String, Integer> getMap() {
		return map;
	}

	public double[][] getEmbeddings() {
		return embeddings;
	}

	public Embeddings() {
	}

	public Embeddings(HashSet<String> lexicon, int sizeEmbeddings, double randomRange) {
		this.numEmbeddings = lexicon.size() + 1;
		this.sizeEmbeddings = sizeEmbeddings;

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
		int count = 0;

		try ( BufferedReader in = new BufferedReader(new FileReader(embeddingsFile)) ) {
			// first pass to count numEmbeddings and sizeEmbeddings
			String line = in.readLine();
			sizeEmbeddings = line.split(" ").length - 1;
			numEmbeddings++;

			while ( in.readLine() != null ) {
				numEmbeddings++;
			}
		} catch ( IOException e ) {
			throw e;
		}

		try ( BufferedReader in = new BufferedReader(new FileReader(embeddingsFile)) ) {
			embeddings = new double[numEmbeddings][sizeEmbeddings];

			// second pass to read embeddings
			String line;
			while ( (line = in.readLine()) != null ) {
				String[] lineSplit = line.split(" ");
				map.put(lineSplit[0], count);

				if ( lineSplit[0].equals(UNK_STRING) ) {
					logger.info("Remapping UNK");
					UNK = count;
				}

				for ( int i = 0; i < sizeEmbeddings; i++ ) {
					embeddings[count][i] = Double.parseDouble(lineSplit[1+i]);
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
					outBuilder.append(" ");
					outBuilder.append(embeddings[i][j]);
				}

				out.println(outBuilder.toString());
			}
		} catch ( IOException e ) {
			throw e;
		}
	}

	private void randomWeights(double randomRange) {
		Random random = new Random();
		for ( int i = 0; i < embeddings.length; i++ ) {
			for ( int j = 0; j < embeddings[i].length; j++ ) {
				embeddings[i][j] = random.nextDouble() * 2 * randomRange - randomRange;
			}
			//normalizeEmbedding(embeddings[i]);
		}
	}

	protected double[] getArray(String key) {
		// method set to private to prevent external mutation of embeddings
		// assumes that callers will not modify the array
		// return actual array instead of clone for performance

		Integer value = map.get(key);
		if ( value != null ) {
			return embeddings[value];
		} else {
			return embeddings[UNK];
		}
	}

	public INDArray getINDArray(String key) {
		return Nd4j.create(getArray(key));
	}

	public void setEmbedding(String key, INDArray embedding) {
		Integer value = map.get(key);
		if ( value != null ) {
			double[] currentEmbedding = embeddings[value];

			for ( int i = 0; i < embedding.length(); i++ ) {
				currentEmbedding[i] = embedding.getDouble(i);
			}

			//normalizeEmbedding(currentEmbedding);
		}
	}

	public void updateEmbedding(String key, INDArray embedding, int offset) {
		Integer value = map.get(key);
		double[] currentEmbedding;

		if ( value != null ) {
			currentEmbedding = embeddings[value];
		} else {
			currentEmbedding = embeddings[UNK];
		}

		for ( int i = 0; i < currentEmbedding.length; i++ ) {
			currentEmbedding[i] -= embedding.getDouble(i + offset);
		}

		//normalizeEmbedding(currentEmbedding);
	}

	public void updateEmbedding(String key, INDArray embedding) {
		updateEmbedding(key, embedding, 0);
	}

	protected void normalizeEmbedding(double[] embedding) {
		// modifies embedding in-place
		double sumOfSquares = 0.0;

		for ( int i = 0; i < embedding.length; i++ ) {
			sumOfSquares += embedding[i] * embedding[i];
		}

		double norm = Math.sqrt(sumOfSquares);

		if ( norm > 0 ) {
			for ( int i = 0; i < embedding.length; i++ ) {
				embedding[i] /= norm;
			}
		}
	}
}
