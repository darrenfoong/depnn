package uk.ac.cam.cl.depnn.io;

import java.util.List;
import java.util.Map;

import joptsimple.OptionParser;
import joptsimple.OptionSet;
import joptsimple.OptionSpec;

public class Params {
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
