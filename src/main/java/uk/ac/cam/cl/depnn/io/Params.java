package uk.ac.cam.cl.depnn.io;

import java.util.List;
import java.util.Map;

import joptsimple.OptionParser;
import joptsimple.OptionSet;
import joptsimple.OptionSpec;

public class Params {
	private final static int W2V_MIN_WORD_FREQUENCY = 5;
	private final static int W2V_ITERATIONS = 1;
	private final static int W2V_LAYER_SIZE = 100;
	private final static int W2V_SEED = 42;
	private final static int W2V_WINDOW_SIZE = 5;

	private final static int NN_NUM_PROPERTIES = 7;
	private final static int NN_BATCH_SIZE = 1000;
	private final static int NN_ITERATIONS = 5;
	private final static int NN_HIDDEN_LAYER_SIZE = 200;
	private final static int NN_SEED = 123;
	private final static double NN_LEARNING_RATE = 1e-6;
	private final static double NN_L1_REG = 1e-1;
	private final static double NN_L2_REG = 2e-4;
	private final static double NN_DROPOUT = 0.5;
	private final static double NN_EMBED_RANDOM_RANGE = 0.01;

	public static OptionParser getBaseOptionParser() {
		OptionParser optionParser = new OptionParser();
		optionParser.accepts("help").forHelp();
		optionParser.accepts("verbose");

		return optionParser;
	}

	public static String printOptions(OptionSet options) {
		StringBuilder outputBuilder = new StringBuilder("Parameters:\n");

		for ( Map.Entry<OptionSpec<?>, List<?>> entry: options.asMap().entrySet() ) {
			if ( !entry.getValue().isEmpty() ) {
				String optionString = entry.getKey().options().get(0);
				String argumentString = entry.getValue().get(0).toString();
				outputBuilder.append(optionString);
				outputBuilder.append(": ");
				outputBuilder.append(argumentString);
				outputBuilder.append("\n");
			}
		}

		return outputBuilder.toString();
	}

	public static OptionParser getTrainNetworkOptionParser() {
		OptionParser optionParser = getBaseOptionParser();

		optionParser.accepts("sentencesFile").withRequiredArg().ofType(String.class).required();
		optionParser.accepts("dependenciesDir").withRequiredArg().ofType(String.class).required();
		optionParser.accepts("modelDir").withRequiredArg().ofType(String.class).required();
		optionParser.accepts("log").withRequiredArg().ofType(String.class).required();

		optionParser.accepts("w2vMinWordFreq").withRequiredArg().ofType(Integer.class).defaultsTo(W2V_MIN_WORD_FREQUENCY);
		optionParser.accepts("w2vIterations").withRequiredArg().ofType(Integer.class).defaultsTo(W2V_ITERATIONS);
		optionParser.accepts("w2vLayerSize").withRequiredArg().ofType(Integer.class).defaultsTo(W2V_LAYER_SIZE);
		optionParser.accepts("w2vSeed").withRequiredArg().ofType(Integer.class).defaultsTo(W2V_SEED);
		optionParser.accepts("w2vWindowSize").withRequiredArg().ofType(Integer.class).defaultsTo(W2V_WINDOW_SIZE);

		optionParser.accepts("nnNumProperties").withRequiredArg().ofType(Integer.class).defaultsTo(NN_NUM_PROPERTIES);
		optionParser.accepts("nnBatchSize").withRequiredArg().ofType(Integer.class).defaultsTo(NN_BATCH_SIZE);
		optionParser.accepts("nnIterations").withRequiredArg().ofType(Integer.class).defaultsTo(NN_ITERATIONS);
		optionParser.accepts("nnHiddenLayerSize").withRequiredArg().ofType(Integer.class).defaultsTo(NN_HIDDEN_LAYER_SIZE);
		optionParser.accepts("nnSeed").withRequiredArg().ofType(Integer.class).defaultsTo(NN_SEED);
		optionParser.accepts("nnLearningRate").withRequiredArg().ofType(Double.class).defaultsTo(NN_LEARNING_RATE);
		optionParser.accepts("nnL1Reg").withRequiredArg().ofType(Double.class).defaultsTo(NN_L1_REG);
		optionParser.accepts("nnL2Reg").withRequiredArg().ofType(Double.class).defaultsTo(NN_L2_REG);
		optionParser.accepts("nnDropout").withRequiredArg().ofType(Double.class).defaultsTo(NN_DROPOUT);
		optionParser.accepts("nnEmbedRandomRange").withRequiredArg().ofType(Double.class).defaultsTo(NN_EMBED_RANDOM_RANGE);

		optionParser.accepts("maxNumBatch").withRequiredArg().ofType(Integer.class).defaultsTo(Integer.MAX_VALUE);

		return optionParser;
	}

	public static OptionParser getTestNetworkOptionParser() {
		OptionParser optionParser = getBaseOptionParser();

		optionParser.accepts("testDir").withRequiredArg().ofType(String.class).required();
		optionParser.accepts("modelDir").withRequiredArg().ofType(String.class).required();
		optionParser.accepts("log").withRequiredArg().ofType(String.class).required();

		return optionParser;
	}
}
