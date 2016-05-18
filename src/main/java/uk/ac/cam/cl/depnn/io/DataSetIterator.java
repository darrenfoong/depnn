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

import uk.ac.cam.cl.depnn.nn.NeuralNetwork;

public class DataSetIterator<T extends NNType> implements Iterator<Pair<DataSet, List<T>>> {
	private final NeuralNetwork<T> network;

	private final RecordReader recordReader;
	private final int batchSize;
	private int correctRecordsPerBatch;
	private int incorrectRecordsPerBatch;

	private final int W2V_LAYER_SIZE;
	private final int NN_NUM_PROPERTIES;
	private final boolean NN_HARD_LABELS;
	private final boolean PRECOMPUTE;

	private ArrayList<T> correctRecords = new ArrayList<T>();
	private ArrayList<T> incorrectRecords = new ArrayList<T>();

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

	public DataSetIterator(NeuralNetwork<T> network, String recordsDir, int batchSize, int W2V_LAYER_SIZE, int NN_NUM_PROPERTIES, boolean NN_HARD_LABELS, boolean PRECOMPUTE, T helper) throws IOException, InterruptedException {
		this.network = network;

		this.recordReader = new CSVRecordReader(0, " ");
		recordReader.initialize(new FileSplit(new File(recordsDir)));

		this.batchSize = batchSize;
		this.W2V_LAYER_SIZE = W2V_LAYER_SIZE;
		this.NN_NUM_PROPERTIES = NN_NUM_PROPERTIES;
		this.NN_HARD_LABELS = NN_HARD_LABELS;
		this.PRECOMPUTE = PRECOMPUTE;

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
			ArrayList<Writable> next = (ArrayList<Writable>) recordReader.next();

			T record = (T) helper.makeRecord(next, NN_HARD_LABELS, catLexicon, slotLexicon, distLexicon, posLexicon);

			if ( record == null ) {
				continue;
			}

			if ( record.getValue() >= 0.5 ) {
				correctRecords.add(record);
			} else {
				incorrectRecords.add(record);
			}
		}

		int numCorrectRecords = correctRecords.size();
		int numIncorrectRecords = incorrectRecords.size();
		int totalRecords = numCorrectRecords + numIncorrectRecords;
		double ratio = ((double) numCorrectRecords)/((double) totalRecords);

		if ( batchSize == 0 ) {
			correctRecordsPerBatch = numCorrectRecords;
			incorrectRecordsPerBatch = numIncorrectRecords;
		} else {
			correctRecordsPerBatch = (int) (ratio * batchSize);
			incorrectRecordsPerBatch = batchSize - correctRecordsPerBatch;
		}

		logger.info("Number of correct records: " + numCorrectRecords);
		logger.info("Number of incorrect records: " + numIncorrectRecords);
		logger.info("Number of correct records per batch: " + correctRecordsPerBatch);
		logger.info("Number of incorrect records per batch: " + incorrectRecordsPerBatch);
		logger.info("All records read");

		reset();

		recordReader.close();
	}

	public void reset() {
		Collections.shuffle(correctRecords);
		Collections.shuffle(incorrectRecords);

		correctIter = correctRecords.iterator();
		incorrectIter = incorrectRecords.iterator();

		nextDataSet = null;
		nextList = null;
		dataSetRead = false;
	}

	private void readDataSet() {
		ArrayList<T> recordsInBatch = new ArrayList<T>();

		for ( int i = 0; i < correctRecordsPerBatch && correctIter.hasNext(); i++ ) {
			recordsInBatch.add(correctIter.next());
		}

		for ( int i = 0; i < incorrectRecordsPerBatch && incorrectIter.hasNext(); i++ ) {
			recordsInBatch.add(incorrectIter.next());
		}

		if ( recordsInBatch.isEmpty() ) {
			nextDataSet = null;
			nextList = null;
			return;
		}

		INDArray records = new NDArray(recordsInBatch.size(), W2V_LAYER_SIZE * helper.getNumProperties());
		INDArray labels = new NDArray(recordsInBatch.size(), 2);

		for ( int i = 0; i < recordsInBatch.size(); i++ ) {
			T record = recordsInBatch.get(i);

			NDArray label = new NDArray(1, 2);
			label.putScalar(0, 1 - record.getValue());
			label.putScalar(1, record.getValue());

			if ( !PRECOMPUTE ) {
				INDArray vector = record.makeVector(network);
				records.putRow(i, vector);
			}

			labels.putRow(i, label);
		}

		try {
			recordReader.close();
		} catch ( IOException e ) {
			logger.error(e);
		}

		nextDataSet = new DataSet(records, labels);
		nextList = recordsInBatch;
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
