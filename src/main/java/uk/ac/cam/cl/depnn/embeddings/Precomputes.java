package uk.ac.cam.cl.depnn.embeddings;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.NDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class Precomputes {
	private int numPrecomputes = 0;
	private int sizePrecomputes = 0;

	/*
	 * assumption: no values are null (for performance i.e. avoid calls to
	 * containKey())
	 */

	private final static Logger logger = LogManager.getLogger(Precomputes.class);

	private HashMap<String, Integer> map;;
	private INDArray precomputes;

	public Precomputes(Embeddings embeddings, INDArray matrix) {
		this.numPrecomputes = embeddings.getNumEmbeddings();
		this.sizePrecomputes = matrix.shape()[1];

		map = embeddings.getMap();
		precomputes = new NDArray(numPrecomputes, sizePrecomputes);

		Iterator<Map.Entry<String, Integer>> iter = map.entrySet().iterator();

		while ( iter.hasNext() ) {
			Map.Entry<String, Integer> next = iter.next();
			precomputes.put(next.getValue(), embeddings.getINDArray(next.getKey()).mmul(matrix));
		}
	}

	public INDArray getINDArray(String key) {
		Integer value = map.get(key);
		return precomputes.get(NDArrayIndex.point(value * sizePrecomputes));
	}
}
