package uk.ac.cam.cl.depnn.io;

import java.util.List;
import java.util.Map;

import joptsimple.OptionParser;
import joptsimple.OptionSet;
import joptsimple.OptionSpec;

public class Params {
	private final static String NN_TYPE = "dep";
	private final static int NN_EPOCHS = 30;
	private final static int NN_SEED = 123;
	private final static int NN_ITERATIONS = 1;
	private final static int NN_BATCH_SIZE = 128;
	private final static int NN_HIDDEN_LAYER_SIZE = 200;
	private final static double NN_LEARNING_RATE = 1e-2;
	private final static double NN_L2_REG = 1e-8;
	private final static double NN_DROPOUT = 0.5;
	private final static double NN_EMBED_RANDOM_RANGE = 0.01;
	private final static boolean NN_HARD_LABELS = true;
	private final static double NN_POS_THRES = 0.8;
	private final static double NN_NEG_THRES = 0.1;

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

		optionParser.accepts("trainDir").withRequiredArg().ofType(String.class).required();
		optionParser.accepts("modelDir").withRequiredArg().ofType(String.class).required();
		optionParser.accepts("log").withRequiredArg().ofType(String.class).required();
		optionParser.accepts("prevModel").withRequiredArg().ofType(String.class).required();

		optionParser.accepts("nnType").withRequiredArg().ofType(String.class).defaultsTo(NN_TYPE);
		optionParser.accepts("nnEpochs").withRequiredArg().ofType(Integer.class).defaultsTo(NN_EPOCHS);
		optionParser.accepts("nnSeed").withRequiredArg().ofType(Integer.class).defaultsTo(NN_SEED);
		optionParser.accepts("nnIterations").withRequiredArg().ofType(Integer.class).defaultsTo(NN_ITERATIONS);
		optionParser.accepts("nnBatchSize").withRequiredArg().ofType(Integer.class).defaultsTo(NN_BATCH_SIZE);
		optionParser.accepts("nnHiddenLayerSize").withRequiredArg().ofType(Integer.class).defaultsTo(NN_HIDDEN_LAYER_SIZE);
		optionParser.accepts("nnLearningRate").withRequiredArg().ofType(Double.class).defaultsTo(NN_LEARNING_RATE);
		optionParser.accepts("nnL2Reg").withRequiredArg().ofType(Double.class).defaultsTo(NN_L2_REG);
		optionParser.accepts("nnDropout").withRequiredArg().ofType(Double.class).defaultsTo(NN_DROPOUT);
		optionParser.accepts("nnEmbedRandomRange").withRequiredArg().ofType(Double.class).defaultsTo(NN_EMBED_RANDOM_RANGE);
		optionParser.accepts("nnHardLabels").withRequiredArg().ofType(Boolean.class).defaultsTo(NN_HARD_LABELS);

		optionParser.accepts("maxNumBatch").withRequiredArg().ofType(Integer.class).defaultsTo(Integer.MAX_VALUE);

		return optionParser;
	}

	public static OptionParser getTestNetworkOptionParser() {
		OptionParser optionParser = getBaseOptionParser();

		optionParser.accepts("testDir").withRequiredArg().ofType(String.class).required();
		optionParser.accepts("modelDir").withRequiredArg().ofType(String.class).required();
		optionParser.accepts("log").withRequiredArg().ofType(String.class).required();

		optionParser.accepts("nnType").withRequiredArg().ofType(String.class).defaultsTo(NN_TYPE);
		optionParser.accepts("nnPosThres").withRequiredArg().ofType(Double.class).defaultsTo(NN_POS_THRES);
		optionParser.accepts("nnNegThres").withRequiredArg().ofType(Double.class).defaultsTo(NN_NEG_THRES);

		return optionParser;
	}

	public static OptionParser getTestSimpleNetworkOptionParser() {
		OptionParser optionParser = getBaseOptionParser();

		optionParser.accepts("testDir").withRequiredArg().ofType(String.class).required();
		optionParser.accepts("modelDir").withRequiredArg().ofType(String.class).required();
		optionParser.accepts("log").withRequiredArg().ofType(String.class).required();

		optionParser.accepts("nnType").withRequiredArg().ofType(String.class).defaultsTo(NN_TYPE);
		optionParser.accepts("nnPosThres").withRequiredArg().ofType(Double.class).defaultsTo(NN_POS_THRES);
		optionParser.accepts("nnNegThres").withRequiredArg().ofType(Double.class).defaultsTo(NN_NEG_THRES);
		optionParser.accepts("precompute").withRequiredArg().ofType(Boolean.class).defaultsTo(true);

		return optionParser;
	}
}
