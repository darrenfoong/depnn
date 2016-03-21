package uk.ac.cam.cl.depnn;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import uk.ac.cam.cl.depnn.embeddings.Embeddings;
import uk.ac.cam.cl.depnn.io.DataSetIterator;
import uk.ac.cam.cl.depnn.io.NNType;
import uk.ac.cam.cl.depnn.utils.ModelUtils;

public class NeuralNetwork<T extends NNType> {
	private int W2V_SEED;
	private int W2V_ITERATIONS;
	private int W2V_BATCH_SIZE;
	private int W2V_LAYER_SIZE;
	private int W2V_WINDOW_SIZE;
	private int W2V_MIN_WORD_FREQUENCY;
	private int W2V_NEGATIVE_SAMPLE;

	private double W2V_LEARNING_RATE;

	private int NN_NUM_PROPERTIES = 7;
	private int NN_EPOCHS;
	private int NN_SEED;
	private int NN_ITERATIONS;
	private int NN_BATCH_SIZE;
	private int NN_HIDDEN_LAYER_SIZE;

	private double NN_LEARNING_RATE;
	private double NN_L2_REG;
	private double NN_DROPOUT;
	private double NN_EMBED_RANDOM_RANGE;
	private boolean NN_HARD_LABELS;

	private int maxNumBatch = Integer.MAX_VALUE;

	private INDArray unkVector;

	public Embeddings catEmbeddings;
	public Embeddings slotEmbeddings;
	public Embeddings distEmbeddings;
	public Embeddings posEmbeddings;

	private WordVectors wordVectors;
	private MultiLayerNetwork network;

	private T helper;

	private final static Logger logger = LogManager.getLogger(NeuralNetwork.class);

	public WordVectors getWordVectors() {
		return wordVectors;
	}

	public MultiLayerNetwork getNetwork() {
		return network;
	}

	// training
	public NeuralNetwork(int w2vSeed,
	                               int w2vIterations,
	                               int w2vBatchSize,
	                               int w2vLayerSize,
	                               int w2vWindowSize,
	                               int w2vMinWordFreq,
	                               int w2vNegativeSample,
	                               double w2vLearningRate,
	                               int nnEpochs,
	                               int nnSeed,
	                               int nnIterations,
	                               int nnBatchSize,
	                               int nnHiddenLayerSize,
	                               double nnLearningRate,
	                               double nnL2Reg,
	                               double nnDropout,
	                               double nnEmbedRandomRange,
	                               boolean nnHardLabels,
	                               int maxNumBatch,
	                               T helper) {
		W2V_SEED = w2vSeed;
		W2V_ITERATIONS = w2vIterations;
		W2V_BATCH_SIZE = w2vBatchSize;
		W2V_LAYER_SIZE = w2vLayerSize;
		W2V_WINDOW_SIZE = w2vWindowSize;
		W2V_MIN_WORD_FREQUENCY = w2vMinWordFreq;
		W2V_NEGATIVE_SAMPLE = w2vNegativeSample;
		W2V_LEARNING_RATE = w2vLearningRate;

		NN_EPOCHS = nnEpochs;
		NN_SEED = nnSeed;
		NN_ITERATIONS = nnIterations;
		NN_BATCH_SIZE = nnBatchSize;
		NN_HIDDEN_LAYER_SIZE = nnHiddenLayerSize;
		NN_LEARNING_RATE = nnLearningRate;
		NN_L2_REG = nnL2Reg;
		NN_DROPOUT = nnDropout;
		NN_EMBED_RANDOM_RANGE = nnEmbedRandomRange;
		NN_HARD_LABELS = nnHardLabels;

		this.maxNumBatch = maxNumBatch;
		this.helper = helper;
	}

	public NeuralNetwork(String prevModelFile,
	                               int nnEpochs,
	                               int nnSeed,
	                               int nnIterations,
	                               int nnBatchSize,
	                               int nnHiddenLayerSize,
	                               double nnLearningRate,
	                               double nnL2Reg,
	                               double nnDropout,
	                               double nnEmbedRandomRange,
	                               boolean nnHardLabels,
	                               int maxNumBatch,
	                               T helper) throws IOException {
		wordVectors = loadWordVectors(prevModelFile);

		NN_EPOCHS = nnEpochs;
		NN_SEED = nnSeed;
		NN_ITERATIONS = nnIterations;
		NN_BATCH_SIZE = nnBatchSize;
		NN_HIDDEN_LAYER_SIZE = nnHiddenLayerSize;
		NN_LEARNING_RATE = nnLearningRate;
		NN_L2_REG = nnL2Reg;
		NN_DROPOUT = nnDropout;
		NN_EMBED_RANDOM_RANGE = nnEmbedRandomRange;
		NN_HARD_LABELS = nnHardLabels;

		this.maxNumBatch = maxNumBatch;
		this.helper = helper;
	}

	// running
	public NeuralNetwork(String modelFile,
	                               String configJsonFile,
	                               String coefficientsFile,
	                               String catEmbeddingsFile,
	                               String slotEmbeddingsFile,
	                               String distEmbeddingsFile,
	                               String posEmbeddingsFile,
	                               T helper) throws IOException {
		wordVectors = loadWordVectors(modelFile);
		network = ModelUtils.loadModelAndParameters(new File(configJsonFile), coefficientsFile);

		catEmbeddings = new Embeddings(catEmbeddingsFile);
		slotEmbeddings = new Embeddings(slotEmbeddingsFile);
		distEmbeddings = new Embeddings(distEmbeddingsFile);
		posEmbeddings = new Embeddings(posEmbeddingsFile);

		this.helper = helper;
	}

	public NeuralNetwork(String modelDir, T helper) throws IOException {
		String modelFile;

		if ( new File(modelDir + "/word2vec.bin").isFile() ) {
			modelFile = modelDir + "/word2vec.bin";
		} else if ( new File(modelDir + "/word2vec.txt").isFile() ) {
			modelFile = modelDir + "/word2vec.txt";
		} else if ( new File(modelDir + "/word2vec.model").isFile() ) {
			modelFile = modelDir + "/word2vec.model";
		} else {
			throw new FileNotFoundException("Missing word2vec model");
		}

		String configJsonFile = modelDir + "/config.json";
		String coefficientsFile = modelDir + "/coeffs";
		String catEmbeddingsFile = modelDir + "/cat.emb";
		String slotEmbeddingsFile = modelDir + "/slot.emb";
		String distEmbeddingsFile = modelDir + "/dist.emb";
		String posEmbeddingsFile = modelDir + "/pos.emb";

		wordVectors = loadWordVectors(modelFile);
		network = ModelUtils.loadModelAndParameters(new File(configJsonFile), coefficientsFile);

		catEmbeddings = new Embeddings(catEmbeddingsFile);
		slotEmbeddings = new Embeddings(slotEmbeddingsFile);
		distEmbeddings = new Embeddings(distEmbeddingsFile);
		posEmbeddings = new Embeddings(posEmbeddingsFile);

		this.helper = helper;
	}

	private void checkUnk(WordVectors wordVectors) {
		if ( wordVectors.hasWord(wordVectors.getUNK()) ) {
			logger.info("wordVectors has UNK");
		} else {
			logger.info("wordVectors does not have UNK");

			String[] otherUnks = { "UNKNOWN", "*UNKNOWN*" };

			for ( String otherUnk : otherUnks ) {
				if ( wordVectors.hasWord(otherUnk) ) {
					logger.info("wordVectors has previous UNK: " + otherUnk);
					logger.info("Remapping UNK");
					wordVectors.setUNK(otherUnk);
					return;
				}
			}

			logger.info("wordVectors still does not have UNK");
			unkVector = Nd4j.zeros(W2V_LAYER_SIZE);
		}
	}

	protected WordVectors loadWordVectors(String modelFile) throws IOException {
		WordVectors w2v = null;

		if ( modelFile.endsWith(".bin") ) {
			w2v = WordVectorSerializer.loadGoogleModel(new File(modelFile), true);
		} else if ( modelFile.endsWith("txt") ) {
			w2v = WordVectorSerializer.loadTxtVectors(new File(modelFile));
		} else {
			w2v = WordVectorSerializer.loadFullModel(modelFile);
		}

		W2V_LAYER_SIZE = w2v.lookupTable().layerSize();
		checkUnk(w2v);

		return w2v;
	}

	public void trainWord2Vec(String sentencesFile) throws FileNotFoundException {
		logger.info("Training word2vec using " + sentencesFile);

		SentenceIterator iter = new BasicLineIterator(sentencesFile);
		TokenizerFactory t = new DefaultTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());

		wordVectors = new Word2Vec.Builder()
				.seed(W2V_SEED)
				.iterations(W2V_ITERATIONS)
				.learningRate(W2V_LEARNING_RATE)
				.batchSize(W2V_BATCH_SIZE)
				.layerSize(W2V_LAYER_SIZE)
				.windowSize(W2V_WINDOW_SIZE)
				.minWordFrequency(W2V_MIN_WORD_FREQUENCY)
				.negativeSample(W2V_NEGATIVE_SAMPLE)
				//.useUnknown(true)
				.iterate(iter)
				.tokenizerFactory(t)
				.build();

		((Word2Vec) wordVectors).fit();

		logger.info("word2vec training complete");
	}

	public void serializeWord2Vec(String modelFile) {
		if ( wordVectors instanceof Word2Vec ) {
			logger.info("Serializing word2vec to " + modelFile);
			WordVectorSerializer.writeFullModel((Word2Vec) wordVectors, modelFile);
		} else {
			logger.error("wordVectors not Word2Vec; cannot serialize");
		}
	}

	public void trainNetwork(String dependenciesDir, String modelDir) throws IOException, InterruptedException {
		logger.info("Training network using " + dependenciesDir);

		int numInput = W2V_LAYER_SIZE * NN_NUM_PROPERTIES;
		int numOutput = 2;

		Nd4j.MAX_SLICES_TO_PRINT = -1;
		Nd4j.MAX_ELEMENTS_PER_SLICE = -1;
		Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

		DataSetIterator<T> iter = new DataSetIterator<T>(this, dependenciesDir, NN_BATCH_SIZE, W2V_LAYER_SIZE, NN_NUM_PROPERTIES, NN_HARD_LABELS, helper);

		catEmbeddings = new Embeddings(iter.getCatLexicon(), W2V_LAYER_SIZE, NN_EMBED_RANDOM_RANGE);
		slotEmbeddings = new Embeddings(iter.getSlotLexicon(), W2V_LAYER_SIZE, NN_EMBED_RANDOM_RANGE);
		distEmbeddings = new Embeddings(iter.getDistLexicon(), W2V_LAYER_SIZE, NN_EMBED_RANDOM_RANGE);
		posEmbeddings = new Embeddings(iter.getPosLexicon(), W2V_LAYER_SIZE, NN_EMBED_RANDOM_RANGE);

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(NN_SEED)
				.iterations(NN_ITERATIONS)
				.learningRate(NN_LEARNING_RATE)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.regularization(true)
				.l2(NN_L2_REG)
				.useDropConnect(true)
				.list(2)
				.layer(0, new DenseLayer.Builder()
								.nIn(numInput)
								.nOut(NN_HIDDEN_LAYER_SIZE)
								.activation("relu")
								.weightInit(WeightInit.XAVIER)
								.updater(Updater.ADAGRAD)
								.dropOut(NN_DROPOUT)
								.build()
				)
				.layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
								.nIn(NN_HIDDEN_LAYER_SIZE)
								.nOut(numOutput)
								.activation("softmax")
								.weightInit(WeightInit.XAVIER)
								.build()
				)
				.pretrain(false)
				.backprop(true)
				.build();

		network = new MultiLayerNetwork(conf);
		network.init();

		network.setListeners(new ScoreIterationListener(NN_ITERATIONS));

		for ( int epochCount = 1; epochCount <= NN_EPOCHS; epochCount++ ) {
			logger.info("Training epoch " + epochCount);

			for ( int batchCount = 1; iter.hasNext() && batchCount <= maxNumBatch; batchCount++ ) {
				logger.info("Training batch " + epochCount + "/" + batchCount);

				Pair<DataSet, List<T>> next = iter.next();
				DataSet trainBatch = next.getFirst();
				List<T> trainList = next.getSecond();

				// trainBatch.normalizeZeroMeanZeroUnitVariance();
				network.fit(trainBatch);

				logger.info("Network updated");

				INDArray epsilon = network.epsilon();

				for ( int i = 0; i < epsilon.rows(); i++ ) {
					INDArray errors = epsilon.getRow(i);

					T record = trainList.get(i);

					record.updateEmbeddings(errors.muli(NN_LEARNING_RATE), W2V_LAYER_SIZE, catEmbeddings, slotEmbeddings, distEmbeddings, posEmbeddings);
				}

				logger.info("Embeddings updated");
			}

			(new File(modelDir + "/epoch" + epochCount + "/")).mkdir();
			serialize(modelDir + "/epoch" + epochCount + "/");

			iter.reset();
		}

		logger.info("Network training complete");
	}

	public void testNetwork(String testDir, String logFile) throws IOException, InterruptedException {
		logger.info("Testing network using " + testDir);

		Evaluation eval = new Evaluation();

		DataSetIterator<T> iter = new DataSetIterator<T>(this, testDir, 0, W2V_LAYER_SIZE, NN_NUM_PROPERTIES, true, helper);
		Pair<DataSet, List<T>> next = iter.next();

		DataSet test = next.getFirst();
		List<T> list = next.getSecond();

		logger.info("Number of test examples: " + test.numExamples());

		INDArray predictions = network.output(test.getFeatures(), false);
		eval.eval(test.getLabels(), predictions);

		logger.info(eval.stats());

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

	public void serializeNetwork(String configJsonFile, String coefficientsFile) throws IOException {
		logger.info("Serializing network to " + configJsonFile + ", " + coefficientsFile);
		ModelUtils.saveModelAndParameters(network, new File(configJsonFile), coefficientsFile);
	}

	public int predict(T example) {
		return network.predict(example.makeVector(this))[0];
	}

	public double predictSoft(T example) {
		return network.output(example.makeVector(this)).getDouble(1);
	}

	public INDArray getWordVector(String word) {
		INDArray vector = wordVectors.getWordVectorMatrixNormalized(word);

		if ( vector == null ) {
			vector = wordVectors.getWordVectorMatrixNormalized(wordVectors.getUNK());

			if ( vector == null ) {
				vector = unkVector;
			}
		}

		// ideally, loadWordVectors() should normalise all the word vectors
		// but stick with this for now

		return vector;
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

	public void serialize(String modelDir) throws IOException {
		String configJsonFile = modelDir + "/config.json";
		String coefficientsFile = modelDir + "/coeffs";
		String catEmbeddingsFile = modelDir + "/cat.emb";
		String slotEmbeddingsFile = modelDir + "/slot.emb";
		String distEmbeddingsFile = modelDir + "/dist.emb";
		String posEmbeddingsFile = modelDir + "/pos.emb";

		serializeNetwork(configJsonFile, coefficientsFile);
		serializeEmbeddings(catEmbeddingsFile, slotEmbeddingsFile, distEmbeddingsFile, posEmbeddingsFile);
	}
}
