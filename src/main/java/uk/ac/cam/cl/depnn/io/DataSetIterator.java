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

import uk.ac.cam.cl.depnn.NeuralNetwork;

public class DataSetIterator<T extends NNType> implements Iterator<Pair<DataSet, List<T>>> {
	private final NeuralNetwork<T> depnn;

	private final RecordReader recordReader;
	private final int batchSize;
	private int correctDepsPerBatch;
	private int incorrectDepsPerBatch;

	private final int W2V_LAYER_SIZE;
	private final int NN_NUM_PROPERTIES;
	private final boolean NN_HARD_LABELS;

	private ArrayList<T> correctDeps = new ArrayList<T>();
	private ArrayList<T> incorrectDeps = new ArrayList<T>();

	private HashSet<String> catLexicon = new HashSet<String>();
	private HashSet<String> slotLexicon = new HashSet<String>();
	private HashSet<String> distLexicon = new HashSet<String>();
	private HashSet<String> posLexicon = new HashSet<String>();

	private DataSet nextDataSet;
	private List<T> nextList;

	private boolean dataSetRead = false;

	private Iterator<T> correctIter;
	private Iterator<T> incorrectIter;

	private T helper;

	public static final Logger logger = LogManager.getLogger(DataSetIterator.class);

	public DataSetIterator(NeuralNetwork depnn, String dependenciesDir, int batchSize, int W2V_LAYER_SIZE, int NN_NUM_PROPERTIES, boolean NN_HARD_LABELS, T helper) throws IOException, InterruptedException {
		this.depnn = depnn;

		this.recordReader = new CSVRecordReader(0, " ");
		recordReader.initialize(new FileSplit(new File(dependenciesDir)));

		this.batchSize = batchSize;
		this.W2V_LAYER_SIZE = W2V_LAYER_SIZE;
		this.NN_NUM_PROPERTIES = NN_NUM_PROPERTIES;
		this.NN_HARD_LABELS = NN_HARD_LABELS;

		this.helper = helper;

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

			T recordList = (T) helper.makeRecord(record, NN_HARD_LABELS, catLexicon, slotLexicon, distLexicon, posLexicon);

			if ( recordList.getValue() >= 0.5 ) {
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
		ArrayList<T> depsInBatch = new ArrayList<T>();

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
			T record = depsInBatch.get(i);

			// make label for labels matrix
			NDArray label = new NDArray(1, 2);
			label.putScalar(0, 1 - record.getValue());
			label.putScalar(1, record.getValue());

			// make dep for deps matrix
			INDArray dep = record.makeVector(depnn);

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
	public Pair<DataSet, List<T>> next() {
		if ( !dataSetRead ) {
			readDataSet();
			dataSetRead = true;
		}

		if ( nextDataSet == null ) {
			throw new NoSuchElementException();
		} else {
			dataSetRead = false;
			return new Pair<DataSet, List<T>>(nextDataSet, nextList);
		}
	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException();
	}
}
