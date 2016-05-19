package uk.ac.cam.cl.depnn.nn;

import java.io.IOException;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;

import uk.ac.cam.cl.depnn.embeddings.Embeddings;
import uk.ac.cam.cl.depnn.embeddings.WordVectors;
import uk.ac.cam.cl.depnn.io.DataSetIterator;
import uk.ac.cam.cl.depnn.io.NNType;
import uk.ac.cam.cl.depnn.io.PrecomputesManager;

public class SimpleNeuralNetwork<T extends NNType> extends NeuralNetwork<T> {
	private SimpleMultiLayerNetwork<T> network;

	private PrecomputesManager<T> manager;

	private final static Logger logger = LogManager.getLogger(SimpleNeuralNetwork.class);

	// training
	public SimpleNeuralNetwork() {
	}

	// running
	public SimpleNeuralNetwork(String modelDir, boolean precompute, T helper) throws IOException {
		String modelFile = modelDir + "/word2vec.txt";

		String coefficientsFile = modelDir + "/coeffs";
		String catEmbeddingsFile = modelDir + "/cat.emb";
		String slotEmbeddingsFile = modelDir + "/slot.emb";
		String distEmbeddingsFile = modelDir + "/dist.emb";
		String posEmbeddingsFile = modelDir + "/pos.emb";

		wordVectors = new WordVectors(modelFile);
		W2V_LAYER_SIZE = wordVectors.getSizeEmbeddings();
		NN_HIDDEN_LAYER_SIZE = 200;
		network = new SimpleMultiLayerNetwork<T>(coefficientsFile, W2V_LAYER_SIZE * helper.getNumProperties(), NN_HIDDEN_LAYER_SIZE, 2);

		catEmbeddings = new Embeddings(catEmbeddingsFile);
		slotEmbeddings = new Embeddings(slotEmbeddingsFile);
		distEmbeddings = new Embeddings(distEmbeddingsFile);
		posEmbeddings = new Embeddings(posEmbeddingsFile);

		this.helper = helper;

		if ( precompute ) {
			manager = new PrecomputesManager<T>(helper, network, W2V_LAYER_SIZE);
			manager.add(wordVectors, 0, true);
			logger.info("0 precomputed");
			manager.add(catEmbeddings, 1, false);
			logger.info("1 precomputed");
			manager.add(slotEmbeddings, 2, false);
			logger.info("2 precomputed");
			manager.add(wordVectors, 3, true);
			logger.info("3 precomputed");
			manager.add(distEmbeddings, 4, false);
			logger.info("4 precomputed");
			manager.add(posEmbeddings, 5, false);
			logger.info("5 precomputed");
			manager.add(posEmbeddings, 6, false);
			logger.info("6 precomputed");
		}
	}

	@Override
	public DataSetIterator<T> genDataSetIterator(String testDir, int batchSize, int W2V_LAYER_SIZE, int NN_NUM_PROPERTIES, int NN_HIDDEN_LAYER_SIZE, boolean NN_HARD_LABELS, T helper) throws IOException, InterruptedException {
		return new DataSetIterator<T>(this, testDir, batchSize, W2V_LAYER_SIZE, NN_NUM_PROPERTIES, NN_HIDDEN_LAYER_SIZE, NN_HARD_LABELS, helper, manager);
	}

	@Override
	public INDArray predict(INDArray inputs, boolean training) {
		if ( manager != null ) {
			return network.outputPrecompute(inputs, training);
		} else {
			return network.output(inputs, training);
		}
	}
}
