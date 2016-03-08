package uk.ac.cam.cl.depnn;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
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

public class DependencyDataSetIterator implements Iterator<Pair<DataSet, List<ArrayList<Writable>>>> {
	private final DependencyNeuralNetwork depnn;

	private final RecordReader recordReader;
	private final int batchSize;
	private int correctDepsPerBatch;
	private int incorrectDepsPerBatch;

	private final int W2V_LAYER_SIZE;
	private final int NN_NUM_PROPERTIES;

	// LinkedLists used here because require only inserts and removes at ends
	private LinkedList<ArrayList<Writable>> correctDeps = new LinkedList<ArrayList<Writable>>();
	private LinkedList<ArrayList<Writable>> incorrectDeps = new LinkedList<ArrayList<Writable>>();

	private HashSet<String> catLexicon = new HashSet<String>();
	private HashSet<String> slotLexicon = new HashSet<String>();
	private HashSet<String> distLexicon = new HashSet<String>();
	private HashSet<String> posLexicon = new HashSet<String>();

	private DataSet nextDataSet;
	private List<ArrayList<Writable>> nextList;

	private boolean dataSetRead = false;

	private Iterator<ArrayList<Writable>> correctIter;
	private Iterator<ArrayList<Writable>> incorrectIter;

	public static final Logger logger = LogManager.getLogger(DependencyDataSetIterator.class);

	public DependencyDataSetIterator(DependencyNeuralNetwork depnn, String dependenciesDir, int batchSize, int W2V_LAYER_SIZE, int NN_NUM_PROPERTIES) throws IOException, InterruptedException {
		this.depnn = depnn;

		this.recordReader = new CSVRecordReader(0, " ");
		recordReader.initialize(new FileSplit(new File(dependenciesDir)));

		this.batchSize = batchSize;
		this.W2V_LAYER_SIZE = W2V_LAYER_SIZE;
		this.NN_NUM_PROPERTIES = NN_NUM_PROPERTIES;

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

			String category = record.get(1).toString();
			String slot = record.get(2).toString();
			String distance = record.get(4).toString();
			String headPos = record.get(5).toString();
			String dependentPos = record.get(6).toString();

			catLexicon.add(category);
			slotLexicon.add(slot);
			distLexicon.add(distance);
			posLexicon.add(headPos);
			posLexicon.add(dependentPos);

			int value = Integer.parseInt(record.get(7).toString());

			if ( value == 0 ) {
				incorrectDeps.add(record);
			} else if ( value == 1 ) {
				correctDeps.add(record);
			} else {
				throw new IllegalArgumentException("Incorrect value");
			}
		}

		// shuffle here

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
		correctIter = correctDeps.iterator();
		incorrectIter = incorrectDeps.iterator();
	}

	private void readDataSet() {
		ArrayList<ArrayList<Writable>> depsInBatch = new ArrayList<ArrayList<Writable>>();

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
			ArrayList<Writable> record = depsInBatch.get(i);

			// head category slot dependent distance head_pos dependent_pos value count
			String head = record.get(0).toString();
			String category = record.get(1).toString();
			String slot = record.get(2).toString();
			String dependent = record.get(3).toString();
			String distance = record.get(4).toString();
			String headPos = record.get(5).toString();
			String dependentPos = record.get(6).toString();

			int value = Integer.parseInt(record.get(7).toString());

			// make label for labels matrix
			NDArray label = new NDArray(1, 2);
			label.putScalar(value, 1);

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
	public Pair<DataSet, List<ArrayList<Writable>>> next() {
		if ( !dataSetRead ) {
			readDataSet();
			dataSetRead = true;
		}

		if ( nextDataSet == null ) {
			throw new NoSuchElementException();
		} else {
			dataSetRead = false;
			return new Pair<DataSet, List<ArrayList<Writable>>>(nextDataSet, nextList);
		}
	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException();
	}
}
