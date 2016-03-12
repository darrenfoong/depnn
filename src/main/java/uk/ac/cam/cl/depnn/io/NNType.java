package uk.ac.cam.cl.depnn.io;

import java.util.ArrayList;
import java.util.HashSet;

import org.canova.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;

import uk.ac.cam.cl.depnn.NeuralNetwork;
import uk.ac.cam.cl.depnn.embeddings.Embeddings;

public abstract class NNType extends ArrayList<String> {
	protected double value;

	public NNType() {
		super();
	}

	public NNType(int size) {
		super(size);
	}

	public double getValue() {
		return value;
	}

	public void setValue(double value) {
		this.value = value;
	}

	public abstract NNType makeRecord(ArrayList<Writable> record,
	                                  boolean hardLabels,
	                                  HashSet<String> catLexicon,
	                                  HashSet<String> slotLexicon,
	                                  HashSet<String> distLexicon,
	                                  HashSet<String> posLexicon);

	public abstract INDArray makeVector(NeuralNetwork  depnn);

	public abstract void updateEmbeddings(INDArray errors,
	                                      int w2vLayerSize,
	                                      Embeddings catEmbeddings,
	                                      Embeddings slotEmbeddings,
	                                      Embeddings distEmbeddings,
	                                      Embeddings posEmbeddings);
 }
