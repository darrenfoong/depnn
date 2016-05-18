package uk.ac.cam.cl.depnn.embeddings;

import java.util.HashMap;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Precomputes {
	private int UNK = 0;

	private boolean words = false;

	/*
	 * assumption: no values are null (for performance i.e. avoid calls to
	 * containKey())
	 */

	private final static Logger logger = LogManager.getLogger(Precomputes.class);

	private HashMap<String, Integer> map;
	private INDArray precomputes;

	public Precomputes(Embeddings embeddings, INDArray matrix, boolean words) {
		this.map = embeddings.getMap();
		this.UNK = embeddings.UNK;
		this.words = words;

		INDArray embeddingsArray = Nd4j.create(embeddings.getEmbeddings());
		precomputes = embeddingsArray.mmul(matrix);
	}

	public INDArray getINDArray(String key) {
		if ( words ) {
			key = key.toLowerCase();
		}

		Integer value = map.get(key);
		if ( value != null ) {
			return precomputes.getRow(value);
		} else {
			return precomputes.getRow(UNK);
		}
	}
}
