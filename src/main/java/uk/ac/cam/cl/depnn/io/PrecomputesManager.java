package uk.ac.cam.cl.depnn.io;

import uk.ac.cam.cl.depnn.embeddings.Embeddings;
import uk.ac.cam.cl.depnn.embeddings.Precomputes;
import uk.ac.cam.cl.depnn.nn.SimpleMultiLayerNetwork;

public class PrecomputesManager<T extends NNType> {
	private int numPrecomputes;
	private int W2V_LAYER_SIZE;
	private SimpleMultiLayerNetwork<T> network;
	private Precomputes[] precomputesList;

	public int getNumPrecomputes() {
		return numPrecomputes;
	}

	public Precomputes getPrecomputes(int i) {
		return precomputesList[i];
	}

	public PrecomputesManager(T helper, SimpleMultiLayerNetwork<T> network, int w2vLayerSize) {
		this.numPrecomputes = helper.getNumProperties();
		this.W2V_LAYER_SIZE = w2vLayerSize;
		this.network = network;
		this.precomputesList = new Precomputes[numPrecomputes];
	}

	public void add(Embeddings embeddings, int i, boolean words) {
		precomputesList[i] = new Precomputes(embeddings, network.getMatrix(i, W2V_LAYER_SIZE), words);
	}
}
