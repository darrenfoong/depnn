package uk.ac.cam.cl.depnn;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.util.ClassPathResource;
import org.canova.api.writable.Writable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import uk.ac.cam.cl.depnn.embeddings.Embeddings;
import uk.ac.cam.cl.depnn.utils.ModelUtils;

public class DependencyNeuralNetwork {
	private final int W2V_MIN_WORD_FREQUENCY = 5;
	private final int W2V_ITERATIONS = 1;
	private final int W2V_LAYER_SIZE = 100;
	private final int W2V_SEED = 42;
	private final int W2V_WINDOW_SIZE = 5;

	private final int NN_NUM_PROPERTIES = 7;
	private final int NN_BATCH_SIZE = 1000;
	private final int NN_ITERATIONS = 5;
	private final int NN_HIDDEN_LAYER_SIZE = 200;
	private final int NN_SEED = 123;
	private final double NN_LEARNING_RATE = 1e-6;
	private final double NN_L1_REG = 1e-1;
	private final double NN_L2_REG = 2e-4;
	private final double NN_DROPOUT = 0.5;

	private Embeddings catEmbeddings;
	private Embeddings slotEmbeddings;
	private Embeddings distEmbeddings;
	private Embeddings posEmbeddings;

	private Word2Vec word2vec;
	private MultiLayerNetwork network;

	// private final static Logger logger = LogManager.getLogger(DependencyNeuralNetwork.class);

	public Word2Vec getWord2Vec() {
		return word2vec;
	}

	public MultiLayerNetwork getNetwork() {
		return network;
	}

	public DependencyNeuralNetwork(String modelFile, String configJsonFile, String coefficientsFile) throws IOException {
		word2vec = WordVectorSerializer.loadFullModel(modelFile);
		network = ModelUtils.loadModelAndParameters(new File(configJsonFile), coefficientsFile);
	}

	public DependencyNeuralNetwork(Word2Vec word2vec, String configJsonFile, String coefficientsFile) throws IOException {
		this.word2vec = word2vec;
		network = ModelUtils.loadModelAndParameters(new File(configJsonFile), coefficientsFile);
	}

	public DependencyNeuralNetwork(String modelFile) {
		word2vec = WordVectorSerializer.loadFullModel(modelFile);
	}

	public DependencyNeuralNetwork() {
	}

	public void trainWord2Vec(String sentencesFile) throws FileNotFoundException {
		String filePath = new ClassPathResource(sentencesFile).getFile().getAbsolutePath();

		SentenceIterator iter = new BasicLineIterator(filePath);
		TokenizerFactory t = new DefaultTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());

		word2vec = new Word2Vec.Builder()
				.minWordFrequency(W2V_MIN_WORD_FREQUENCY)
				.iterations(W2V_ITERATIONS)
				.layerSize(W2V_LAYER_SIZE)
				.seed(W2V_SEED)
				.windowSize(W2V_WINDOW_SIZE)
				.iterate(iter)
				.tokenizerFactory(t)
				.build();

		word2vec.fit();
	}

	public void serializeWord2Vec(String modelFile) {
		WordVectorSerializer.writeFullModel(word2vec, modelFile);
	}

	public void trainNetwork(String dependenciesDir) throws IOException, InterruptedException {
		int numInput = W2V_LAYER_SIZE * NN_NUM_PROPERTIES;
		int numOutput = 2;

		Nd4j.MAX_SLICES_TO_PRINT = -1;
		Nd4j.MAX_ELEMENTS_PER_SLICE = -1;
		Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

		DependencyDataSetIterator iter = new DependencyDataSetIterator(this, dependenciesDir, NN_BATCH_SIZE, W2V_LAYER_SIZE, NN_NUM_PROPERTIES);

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(NN_SEED)
				.iterations(NN_ITERATIONS)
				.learningRate(NN_LEARNING_RATE)
				.optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
				.l1(NN_L1_REG)
				.regularization(true)
				.l2(NN_L2_REG)
				.useDropConnect(true)
				.list(2)
				.layer(0, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
								.nIn(numInput)
								.nOut(NN_HIDDEN_LAYER_SIZE)
								.weightInit(WeightInit.XAVIER)
								.k(1) // # contrastive divergence iterations
								.activation("relu")
								.lossFunction(LossFunctions.LossFunction.RMSE_XENT)
								.updater(Updater.ADAGRAD)
								.dropOut(NN_DROPOUT)
								.build()
				)
				.layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
								.nIn(NN_HIDDEN_LAYER_SIZE)
								.nOut(numOutput)
								.activation("softmax")
								.build()
				)
				.build();

		network = new MultiLayerNetwork(conf);
		network.init();

		while ( iter.hasNext() ) {
			DataSet trainBatch = iter.next();
			trainBatch.normalizeZeroMeanZeroUnitVariance();
			network.fit(trainBatch);

			Gradient gradient = network.gradient();
			// update non-word embeddings
		}
	}

	public void serializeNetwork(String configJsonFile, String coefficientsFile) throws IOException {
		ModelUtils.saveModelAndParameters(network, new File(configJsonFile), coefficientsFile);
	}

	public int predict(String head,
                       String category,
                       String slot,
                       String dependent,
                       String distance,
                       String headPos,
                       String dependentPos) {
		return network.predict(makeVector(head,
									category,
									slot,
									dependent,
									distance,
									headPos,
									dependentPos))[0];
	}

	protected INDArray makeVector(String head,
	                            String category,
	                            String slot,
	                            String dependent,
	                            String distance,
	                            String headPos,
	                            String dependentPos) {
		INDArray headVector = word2vec.getWordVectorMatrix(head);
		INDArray dependentVector = word2vec.getWordVectorMatrix(dependent);

		// to use another embedding space
		INDArray categoryVector = catEmbeddings.getINDArray(category);
		INDArray slotVector = slotEmbeddings.getINDArray(slot);
		INDArray distanceVector = distEmbeddings.getINDArray(distance);
		INDArray headPosVector = posEmbeddings.getINDArray(headPos);
		INDArray dependentPosVector= posEmbeddings.getINDArray(dependentPos);

		return Nd4j.concat(1, headVector,
							categoryVector,
							slotVector,
							dependentVector,
							distanceVector,
							headPosVector,
							dependentPosVector);
	}
}
