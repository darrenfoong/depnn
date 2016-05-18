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

public class SimpleNeuralNetwork<T extends NNType> extends NeuralNetwork<T> {
	private WordVectors wordVectors;
	private SimpleMultiLayerNetwork network;

	private final static Logger logger = LogManager.getLogger(SimpleNeuralNetwork.class);

	// training
	public SimpleNeuralNetwork() {
	}

	// running
	public SimpleNeuralNetwork(String modelDir, T helper) throws IOException {
		String modelFile = modelDir + "/word2vec.txt";

		// String configJsonFile = modelDir + "/config.json";
		String coefficientsFile = modelDir + "/coeffs";
		String catEmbeddingsFile = modelDir + "/cat.emb";
		String slotEmbeddingsFile = modelDir + "/slot.emb";
		String distEmbeddingsFile = modelDir + "/dist.emb";
		String posEmbeddingsFile = modelDir + "/pos.emb";

		wordVectors = new WordVectors(modelFile);
		W2V_LAYER_SIZE = wordVectors.getSizeEmbeddings();
		network = new SimpleMultiLayerNetwork(coefficientsFile, W2V_LAYER_SIZE * helper.getNumProperties(), 200, 2);

		catEmbeddings = new Embeddings(catEmbeddingsFile);
		slotEmbeddings = new Embeddings(slotEmbeddingsFile);
		distEmbeddings = new Embeddings(distEmbeddingsFile);
		posEmbeddings = new Embeddings(posEmbeddingsFile);

		this.helper = helper;
	}

	@Override
	public void testNetwork(String testDir, String logFile, double posThres, double negThres) throws IOException, InterruptedException {
		logger.info("Testing network using " + testDir);

		DataSetIterator<T> iter = new DataSetIterator<T>(this, testDir, 0, W2V_LAYER_SIZE, helper.getNumProperties(), true, helper);
		Pair<DataSet, List<T>> next = iter.next();

		DataSet test = next.getFirst();
		List<T> list = next.getSecond();

		logger.info("Number of test examples: " + test.numExamples());

		INDArray predictions = network.output(test.getFeatures(), false);

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

		logger.info("Network testing complete");
	}

	@Override
	public INDArray getWordVector(String word) {
		return wordVectors.getINDArray(word.toLowerCase());
	}
}
