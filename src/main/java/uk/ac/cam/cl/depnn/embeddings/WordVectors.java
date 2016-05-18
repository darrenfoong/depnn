package uk.ac.cam.cl.depnn.embeddings;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class WordVectors extends Embeddings {
	private final static String UNK_STRING = "UNKNOWN";

	/*
	 * assumption: no values are null (for performance i.e. avoid calls to
	 * containKey())
	 */

	private final static Logger logger = LogManager.getLogger(WordVectors.class);

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
}
