package uk.ac.cam.cl.depnn.io;

import java.util.ArrayList;
import java.util.HashSet;

import org.canova.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import uk.ac.cam.cl.depnn.embeddings.Embeddings;
import uk.ac.cam.cl.depnn.nn.NeuralNetwork;

public class Feature extends NNType {
	private int sigmoidScaleFactor = 20;

	@Override
	public int getNumProperties() {
		return 9;
	}

	@Override
	public NNType makeRecord(ArrayList<Writable> record, boolean hardLabels, HashSet<String> catLexicon, HashSet<String> slotLexicon, HashSet<String> distLexicon, HashSet<String> posLexicon) {
		Feature result = new Feature();

		String topCat = record.get(0).toString();
		String leftCat = record.get(1).toString();
		String rightCat = record.get(2).toString();
		String topCatWord = record.get(3).toString();
		String leftCatWord = record.get(4).toString();
		String rightCatWord = record.get(5).toString();
		String topCatPos = record.get(6).toString();
		String leftCatPos = record.get(7).toString();
		String rightCatPos = record.get(8).toString();

		String valueString;

		if ( hardLabels ) {
			valueString = record.get(9).toString();
			result.value = Double.parseDouble(valueString);
		} else {
			valueString = record.get(10).toString();
			result.value = Math.tanh(Double.parseDouble(valueString) / sigmoidScaleFactor);
		}

		catLexicon.add(topCat);
		catLexicon.add(leftCat);
		catLexicon.add(rightCat);
		posLexicon.add(topCatPos);
		posLexicon.add(leftCatPos);
		posLexicon.add(rightCatPos);

		result.add(topCat);
		result.add(leftCat);
		result.add(rightCat);
		result.add(topCatWord);
		result.add(leftCatWord);
		result.add(rightCatWord);
		result.add(topCatPos);
		result.add(leftCatPos);
		result.add(rightCatPos);

		return result;
	}

	@Override
	public INDArray makeVector(NeuralNetwork<? extends NNType> network) {
		String topCat = this.get(0);
		String leftCat = this.get(1);
		String rightCat = this.get(2);
		String topCatWord = this.get(3);
		String leftCatWord = this.get(4);
		String rightCatWord = this.get(5);
		String topCatPos = this.get(6);
		String leftCatPos = this.get(7);
		String rightCatPos = this.get(8);

		INDArray topCatVector = network.catEmbeddings.getINDArray(topCat);
		INDArray leftCatVector = network.catEmbeddings.getINDArray(leftCat);
		INDArray rightCatVector = network.catEmbeddings.getINDArray(rightCat);
		INDArray topCatWordVector = network.getWordVector(topCatWord);
		INDArray leftCatWordVector = network.getWordVector(leftCatWord);
		INDArray rightCatWordVector = network.getWordVector(rightCatWord);
		INDArray topCatPosVector = network.posEmbeddings.getINDArray(topCatPos);
		INDArray leftCatPosVector = network.posEmbeddings.getINDArray(leftCatPos);
		INDArray rightCatPosVector = network.posEmbeddings.getINDArray(rightCatPos);

		return Nd4j.concat(1,
							topCatVector,
							leftCatVector,
							rightCatVector,
							topCatWordVector,
							leftCatWordVector,
							rightCatWordVector,
							topCatPosVector,
							leftCatPosVector,
							rightCatPosVector);
	}

	@Override
	public void updateEmbeddings(INDArray errors, int w2vLayerSize, Embeddings catEmbeddings, Embeddings slotEmbeddings, Embeddings distEmbeddings, Embeddings posEmbeddings) {
		catEmbeddings.updateEmbedding(this.get(0), errors, 0 * w2vLayerSize);
		catEmbeddings.updateEmbedding(this.get(1), errors, 1 * w2vLayerSize);
		catEmbeddings.updateEmbedding(this.get(2), errors, 2 * w2vLayerSize);
		posEmbeddings.updateEmbedding(this.get(14), errors , 6 * w2vLayerSize);
		posEmbeddings.updateEmbedding(this.get(15), errors , 7 * w2vLayerSize);
		posEmbeddings.updateEmbedding(this.get(16), errors , 8 * w2vLayerSize);
	}
}
