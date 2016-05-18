package uk.ac.cam.cl.depnn.embeddings;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class WordVectors {
	private int numEmbeddings = 0;
	private int sizeEmbeddings = 0;
	private int UNK = 0;
	private final static String UNK_STRING = "UNKNOWN";

	/*
	 * assumption: no values are null (for performance i.e. avoid calls to
	 * containKey())
	 */

	private final static Logger logger = LogManager.getLogger(WordVectors.class);

	private HashMap<String, Integer> map = new HashMap<String, Integer>();
	private double[][] embeddings;

	public int getSizeEmbeddings() {
		return sizeEmbeddings;
	}

	public WordVectors(String embeddingsFile) throws IOException {
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

				normalizeEmbedding(embeddings[count]);

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

	private double[] getArray(String key) {
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

	private void normalizeEmbedding(double[] embedding) {
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
