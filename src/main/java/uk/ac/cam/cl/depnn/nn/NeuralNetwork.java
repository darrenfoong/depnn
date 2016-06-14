package uk.ac.cam.cl.depnn.nn;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;

import uk.ac.cam.cl.depnn.embeddings.Embeddings;
import uk.ac.cam.cl.depnn.embeddings.WordVectors;
import uk.ac.cam.cl.depnn.io.DataSetIterator;
import uk.ac.cam.cl.depnn.io.NNType;

public class NeuralNetwork<T extends NNType> {
	protected int W2V_LAYER_SIZE;
	protected int NN_HIDDEN_LAYER_SIZE;

	public Embeddings catEmbeddings;
	public Embeddings slotEmbeddings;
	public Embeddings distEmbeddings;
	public Embeddings posEmbeddings;

	protected WordVectors wordVectors;
	private MultiLayerNetwork network;

	protected T helper;

	private final static Logger logger = LogManager.getLogger(NeuralNetwork.class);

	public WordVectors getWordVectors() {
		return wordVectors;
	}

	public MultiLayerNetwork getNetwork() {
		return network;
	}

	public void testNetwork(String testDir, String logFile, double posThres, double negThres) throws IOException, InterruptedException {
		logger.info("Testing network using " + testDir);
		long start = System.nanoTime();

		DataSetIterator<T> iter = genDataSetIterator(testDir, 0, W2V_LAYER_SIZE, helper.getNumProperties(), NN_HIDDEN_LAYER_SIZE, true, helper);
		Pair<DataSet, List<T>> next = iter.next();

		DataSet test = next.getFirst();
		List<T> list = next.getSecond();

		logger.info("Number of test examples: " + test.numExamples());

		long startP = System.nanoTime();
		logger.info("Load time: " + (startP-start));

		INDArray predictions = predict(test.getFeatures(), false);

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

		logger.info("Network testing complete");

		long end = System.nanoTime();
		logger.info("Total time: " + (end-start));
	}

	protected void evaluateThresholds(INDArray labels, INDArray predictions, double posThres, double negThres) {
		for ( int j = 5; j < 10; j++ ) {
			double posThreshold = j / ((double) 10);
			double negThreshold = (10 - j) / ((double) 10);

			evaluateThreshold(labels, predictions, posThreshold, negThreshold);
		}

		evaluateThreshold(labels, predictions, posThres, negThres);
	}

	private void evaluateThreshold(INDArray labels, INDArray predictions, double posThreshold, double negThreshold) {
		Evaluation eval = new Evaluation();

		ArrayList<INDArray> sublabelsList = new ArrayList<INDArray>();
		ArrayList<INDArray> subpredictionsList = new ArrayList<INDArray>();

		for ( int i = 0; i < labels.size(0); i++ ) {
			double prediction = predictions.getDouble(i, 1);

			if ( ( prediction >= posThreshold ) || ( prediction <= negThreshold ) ){
				sublabelsList.add(labels.getRow(i));
				subpredictionsList.add(predictions.getRow(i));
			}
		}

		INDArray sublabels = new NDArray(sublabelsList.size() ,2);
		INDArray subpredictions = new NDArray(subpredictionsList.size() ,2);

		for ( int i = 0; i < sublabelsList.size(); i++ ) {
			sublabels.putRow(i, sublabelsList.get(i));
			subpredictions.putRow(i, subpredictionsList.get(i));
		}

		eval.eval(sublabels, subpredictions);

		logger.info("Evaluation threshold: " + posThreshold + ", " + negThreshold);
		logger.info(eval.stats());
		logger.info("");
	}

	public DataSetIterator<T> genDataSetIterator(String testDir, int NN_BATCH_SIZE, int W2V_LAYER_SIZE, int NN_NUM_PROPERTIES, int NN_HIDDEN_LAYER_SIZE, boolean NN_HARD_LABELS, T helper) throws IOException, InterruptedException {
		return new DataSetIterator<T>(this, testDir, NN_BATCH_SIZE, W2V_LAYER_SIZE, NN_NUM_PROPERTIES, NN_HIDDEN_LAYER_SIZE, NN_HARD_LABELS, helper, null);
	}

	public INDArray predict(INDArray inputs, boolean training) {
		return network.output(inputs, training);
	}

	public double predictSoft(T example) {
		return network.output(example.makeVector(this)).getDouble(1);
	}

	public double[] predictSoft(ArrayList<T> examples) {
		INDArray vectors = new NDArray(examples.size(), W2V_LAYER_SIZE * helper.getNumProperties());

		for ( int i = 0; i < examples.size(); i++ ) {
			T example = examples.get(i);

			INDArray vector = example.makeVector(this);

			vectors.putRow(i, vector);
		}

		INDArray predictions = predict(vectors, false);

		double[] res = new double[examples.size()];

		for ( int i = 0; i < examples.size(); i++ ) {
			double prediction = predictions.getDouble(i, 1);
			res[i] = prediction;
		}

		return res;
	}

	public double[] predict(ArrayList<T> examples, double posThres, double negThres) {
		double[] predictions = predictSoft(examples);
		double[] res = new double[examples.size()];

		for ( int i = 0; i < examples.size(); i++ ) {
			double prediction = predictions[i];

			if ( prediction >= posThres ) {
				res[i] = 1;
			} else if ( prediction <= negThres ) {
				res[i] = -1;
			} else {
				res[i] = 0;
			}
		}

		return res;
	}

	public double predictSum(ArrayList<T> examples, double posThres, double negThres) {
		double[] predictions = predict(examples, posThres, negThres);

		double res = 0;

		for ( double prediction : predictions ) {
			res += prediction;
		}

		return res;
	}

	public INDArray getWordVector(String word) {
		return wordVectors.getINDArray(word.toLowerCase());
	}

	public void serializeEmbeddings(String catEmbeddingsFile,
	                                String slotEmbeddingsFile,
	                                String distEmbeddingsFile,
	                                String posEmbeddingsFile) throws IOException {
		logger.info("Serializing embeddings to " + catEmbeddingsFile + ", "
												 + slotEmbeddingsFile + ", "
												 + distEmbeddingsFile + ", "
												 + posEmbeddingsFile);
		catEmbeddings.serializeEmbeddings(catEmbeddingsFile);
		slotEmbeddings.serializeEmbeddings(slotEmbeddingsFile);
		distEmbeddings.serializeEmbeddings(distEmbeddingsFile);
		posEmbeddings.serializeEmbeddings(posEmbeddingsFile);
	}
}
