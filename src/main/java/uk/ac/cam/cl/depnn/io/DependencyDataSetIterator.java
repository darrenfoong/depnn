package uk.ac.cam.cl.depnn.io;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.writable.Writable;
import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;

import uk.ac.cam.cl.depnn.DependencyNeuralNetwork;

public class DependencyDataSetIterator implements Iterator<Pair<DataSet, List<ArrayList<String>>>> {
	private final DependencyNeuralNetwork depnn;

	private final RecordReader recordReader;
	private final int batchSize;
	private int correctDepsPerBatch;
	private int incorrectDepsPerBatch;

	private final int W2V_LAYER_SIZE;
	private final int NN_NUM_PROPERTIES;
	private final boolean NN_HARD_LABELS;

	private int sigmoidScaleFactor = 20;

	private ArrayList<ArrayList<String>> correctDeps = new ArrayList<ArrayList<String>>();
	private ArrayList<ArrayList<String>> incorrectDeps = new ArrayList<ArrayList<String>>();

	private HashSet<String> catLexicon = new HashSet<String>();
	private HashSet<String> slotLexicon = new HashSet<String>();
	private HashSet<String> distLexicon = new HashSet<String>();
	private HashSet<String> posLexicon = new HashSet<String>();

	private DataSet nextDataSet;
	private List<ArrayList<String>> nextList;

	private boolean dataSetRead = false;

	private Iterator<ArrayList<String>> correctIter;
	private Iterator<ArrayList<String>> incorrectIter;

	public static final Logger logger = LogManager.getLogger(DependencyDataSetIterator.class);

	public DependencyDataSetIterator(DependencyNeuralNetwork depnn, String dependenciesDir, int batchSize, int W2V_LAYER_SIZE, int NN_NUM_PROPERTIES, boolean NN_HARD_LABELS) throws IOException, InterruptedException {
		this.depnn = depnn;

		this.recordReader = new CSVRecordReader(0, " ");
		recordReader.initialize(new FileSplit(new File(dependenciesDir)));

		this.batchSize = batchSize;
		this.W2V_LAYER_SIZE = W2V_LAYER_SIZE;
		this.NN_NUM_PROPERTIES = NN_NUM_PROPERTIES;
		this.NN_HARD_LABELS = NN_HARD_LABELS;

		readAll();
	}

	public HashSet<String> getCatLexicon() {
		return catLexicon;
	}

	public HashSet<String> getSlotLexicon() {
		return slotLexicon;
	}

	public HashSet<String> getDistLexicon() {
		return distLexicon;
	}

	public HashSet<String> getPosLexicon() {
		return posLexicon;
	}

	private void readAll() throws IOException {
		while ( recordReader.hasNext() ) {
			ArrayList<Writable> record = (ArrayList<Writable>) recordReader.next();

			String head = record.get(0).toString();
			String category = record.get(1).toString();
			String slot = record.get(2).toString();
			String dependent = record.get(3).toString();
			String distance = record.get(4).toString();
			String headPos = record.get(5).toString();
			String dependentPos = record.get(6).toString();

			String valueString;
			double value;

			if ( NN_HARD_LABELS ) {
				valueString = record.get(7).toString();
				value = Double.parseDouble(valueString);
			} else {
				valueString = record.get(8).toString();
				value = Math.tanh(Double.parseDouble(valueString) / sigmoidScaleFactor);
			}

			catLexicon.add(category);
			slotLexicon.add(slot);
			distLexicon.add(distance);
			posLexicon.add(headPos);
			posLexicon.add(dependentPos);

			ArrayList<String> recordList = new ArrayList<String>(8);
			recordList.add(head);
			recordList.add(category);
			recordList.add(slot);
			recordList.add(dependent);
			recordList.add(distance);
			recordList.add(headPos);
			recordList.add(dependentPos);
			recordList.add(valueString);

			if ( value >= 0.5 ) {
				correctDeps.add(recordList);
			} else {
				incorrectDeps.add(recordList);
			}
		}

		int numCorrectDeps = correctDeps.size();
		int numIncorrectDeps = incorrectDeps.size();
		int totalDeps = numCorrectDeps + numIncorrectDeps;
		double ratio = ((double) numCorrectDeps)/((double) totalDeps);

		if ( batchSize == 0 ) {
			correctDepsPerBatch = numCorrectDeps;
			incorrectDepsPerBatch = numIncorrectDeps;
		} else {
			correctDepsPerBatch = (int) (ratio * batchSize);
			incorrectDepsPerBatch = batchSize - correctDepsPerBatch;
		}

		logger.info("Number of correct deps: " + numCorrectDeps);
		logger.info("Number of incorrect deps: " + numIncorrectDeps);
		logger.info("Number of correct deps per batch: " + correctDepsPerBatch);
		logger.info("Number of incorrect deps per batch: " + incorrectDepsPerBatch);
		logger.info("All deps read");

		reset();

		recordReader.close();
	}

	public void reset() {
		Collections.shuffle(correctDeps);
		Collections.shuffle(incorrectDeps);

		correctIter = correctDeps.iterator();
		incorrectIter = incorrectDeps.iterator();

		nextDataSet = null;
		nextList = null;
		dataSetRead = false;
	}

	private void readDataSet() {
		ArrayList<ArrayList<String>> depsInBatch = new ArrayList<ArrayList<String>>();

		for ( int i = 0; i < correctDepsPerBatch && correctIter.hasNext(); i++ ) {
			depsInBatch.add(correctIter.next());
		}

		for ( int i = 0; i < incorrectDepsPerBatch && incorrectIter.hasNext(); i++ ) {
			depsInBatch.add(incorrectIter.next());
		}

		if ( depsInBatch.isEmpty() ) {
			nextDataSet = null;
			nextList = null;
			return;
		}

		INDArray deps = new NDArray(depsInBatch.size(), W2V_LAYER_SIZE * NN_NUM_PROPERTIES);
		INDArray labels = new NDArray(depsInBatch.size(), 2);

		for ( int i = 0; i < depsInBatch.size(); i++ ) {
			ArrayList<String> record = depsInBatch.get(i);

			// head category slot dependent distance head_pos dependent_pos value count
			String head = record.get(0);
			String category = record.get(1);
			String slot = record.get(2);
			String dependent = record.get(3);
			String distance = record.get(4);
			String headPos = record.get(5);
			String dependentPos = record.get(6);

			double value = Double.parseDouble(record.get(7));

			// make label for labels matrix
			NDArray label = new NDArray(1, 2);
			label.putScalar(0, 1 - value);
			label.putScalar(1, value);

			// make dep for deps matrix
			INDArray dep = depnn.makeVector(head, category, slot, dependent, distance, headPos, dependentPos);

			deps.putRow(i, dep);
			labels.putRow(i, label);
		}

		try {
			recordReader.close();
		} catch ( IOException e ) {
			logger.error(e);
		}

		nextDataSet = new DataSet(deps, labels);
		nextList = depsInBatch;
	}

	@Override
	public boolean hasNext() {
		if ( !dataSetRead ) {
			readDataSet();
			dataSetRead = true;
		}

		return nextDataSet != null;
	}

	@Override
	public Pair<DataSet, List<ArrayList<String>>> next() {
		if ( !dataSetRead ) {
			readDataSet();
			dataSetRead = true;
		}

		if ( nextDataSet == null ) {
			throw new NoSuchElementException();
		} else {
			dataSetRead = false;
			return new Pair<DataSet, List<ArrayList<String>>>(nextDataSet, nextList);
		}
	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException();
	}
}
