package uk.ac.cam.cl.depnn.nn;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

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
	public SimpleNeuralNetwork(String modelDir, T helper) throws IOException {
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
	}

	public void testNetwork(String testDir, String logFile, double posThres, double negThres, boolean precompute) throws IOException, InterruptedException {
		logger.info("Testing network using " + testDir);
		long start = System.nanoTime();

		DataSetIterator<T> iter;

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

			iter = new DataSetIterator<T>(this, testDir, 0, W2V_LAYER_SIZE, helper.getNumProperties(), NN_HIDDEN_LAYER_SIZE, true, helper, manager);
		} else {
			iter = new DataSetIterator<T>(this, testDir, 0, W2V_LAYER_SIZE, helper.getNumProperties(), NN_HIDDEN_LAYER_SIZE, true, helper);
		}

		Pair<DataSet, List<T>> next = iter.next();

		DataSet test = next.getFirst();
		List<T> list = next.getSecond();

		logger.info("Number of test examples: " + test.numExamples());

		INDArray predictions;

		long startP = System.nanoTime();
		logger.info("Load time: " + (startP-start));
		if ( precompute ) {
			predictions = network.outputPrecompute(test.getFeatures(), false);
		} else {
			predictions = network.output(test.getFeatures(), false);
		}
		long endP = System.nanoTime();
		logger.info("Predict time: " + (endP-startP));

		evaluateThresholds(test.getLabels(), predictions, posThres, negThres);

		try ( PrintWriter outCorrect = new PrintWriter(new BufferedWriter(new FileWriter(logFile + ".classified1")));
				PrintWriter outIncorrect = new PrintWriter(new BufferedWriter(new FileWriter(logFile + ".classified0"))) ) {
			logger.info("Writing to files");

			for ( int i = 0; i < list.size(); i++ ) {
				String example = list.get(i).toString();
				double prediction = predictions.getDouble(i, 1);

				if ( prediction >= 0.5 ) {
					outCorrect.println(example);
				} else {
					outIncorrect.println(example);
				}
			}
		} catch ( FileNotFoundException e ) {
			logger.error(e);
		} catch ( IOException e ) {
			logger.error(e);
		}

		long end = System.nanoTime();
		logger.info("Network testing complete");
		logger.info("Time: " + (end-start));
	}
}
