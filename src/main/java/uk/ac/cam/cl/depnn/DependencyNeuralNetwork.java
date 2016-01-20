package uk.ac.cam.cl.depnn;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

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

import uk.ac.cam.cl.depnn.utils.ModelUtils;

public class DependencyNeuralNetwork {
	private final int W2V_MIN_WORD_FREQUENCY = 5;
	private final int W2V_ITERATIONS = 1;
	private final int W2V_LAYER_SIZE = 100;
	private final int W2V_SEED = 42;
	private final int W2V_WINDOW_SIZE = 5;

	private final int NN_ITERATIONS = 5;
	private final int NN_SEED = 123;

	private Word2Vec word2vec;
	private MultiLayerNetwork network;

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

	public void trainNetwork(String dependenciesDir) {
		int numInput = word2vec.getLayerSize() * 2;
		int numOutput = 2;

		Nd4j.MAX_SLICES_TO_PRINT = -1;
		Nd4j.MAX_ELEMENTS_PER_SLICE = -1;

		DataSet train = null;

		try {
			train = importData(dependenciesDir);
		} catch (Exception e) {
			System.err.println(e);
			// implement better exception handling!
			return;
		}

		train.normalizeZeroMeanZeroUnitVariance();
		Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(NN_SEED) // Locks in weight initialization for tuning
				.iterations(NN_ITERATIONS) // # training iterations predict/classify & backprop
				.learningRate(1e-6f) // Optimization step size
				.optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT) // Backprop to calculate gradients
				.l1(1e-1).regularization(true).l2(2e-4)
				.useDropConnect(true)
				.list(2) // # NN layers (doesn't count input layer)
				.layer(0, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
								.nIn(numInput) // # input nodes
								.nOut(3) // # fully connected hidden layer nodes. Add list if multiple layers.
								.weightInit(WeightInit.XAVIER) // Weight initialization
								.k(1) // # contrastive divergence iterations
								.activation("relu") // Activation function type
								.lossFunction( LossFunctions.LossFunction.RMSE_XENT) // Loss function type
								.updater(Updater.ADAGRAD)
								.dropOut(0.5)
								.build()
				) // NN layer type
				.layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
								.nIn(3) // # input nodes
								.nOut(numOutput) // # output nodes
								.activation("softmax")
								.build()
				) // NN layer type
				.build();

		network = new MultiLayerNetwork(conf);
		network.init();

		network.fit(train);
	}

	public void serializeNetwork(String configJsonFile, String coefficientsFile) throws IOException {
		ModelUtils.saveModelAndParameters(network, new File(configJsonFile), coefficientsFile);
	}

	public int predict(String head, String dependent) {
		return network.predict(makeVector(head, dependent))[0];
	}

	private INDArray makeVector(String head, String dependent) {
		INDArray headVector = word2vec.getWordVectorMatrix(head);
		INDArray dependentVector = word2vec.getWordVectorMatrix(dependent);

		return Nd4j.concat(1, headVector, dependentVector);
	}

	private DataSet importData(String dependenciesDir) throws FileNotFoundException, IOException, InterruptedException {
		RecordReader recordReader = new CSVRecordReader(0, " ");
		recordReader.initialize(new FileSplit(new ClassPathResource(dependenciesDir).getFile()));

		int numRecords = 0;

		while ( recordReader.hasNext() ) {
			recordReader.next();
			numRecords++;
		}

		INDArray deps = new NDArray(numRecords, word2vec.getLayerSize() * 2);
		INDArray labels = new NDArray(numRecords, 2);

		int i = 0;

		while ( recordReader.hasNext() ) {
			ArrayList<Writable> record = (ArrayList<Writable>) recordReader.next();

			String head = record.get(0).toString();
			String dependent = record.get(1).toString();
			int value = Integer.parseInt(record.get(2).toString());

			NDArray label = new NDArray(1, 2);
			label.putScalar(value, 1);

			INDArray dep = makeVector(head, dependent);

			deps.putRow(i, dep);
			labels.putRow(i, label);
			i++;
		}

		recordReader.close();

		return new DataSet(deps, labels);
	}
}
