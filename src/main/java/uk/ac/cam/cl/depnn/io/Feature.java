package uk.ac.cam.cl.depnn.io;

import java.util.ArrayList;
import java.util.HashSet;

import org.canova.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import uk.ac.cam.cl.depnn.NeuralNetwork;
import uk.ac.cam.cl.depnn.embeddings.Embeddings;

public class Feature extends NNType {
	private int sigmoidScaleFactor = 20;

	@Override
	public NNType makeRecord(ArrayList<Writable> record, boolean hardLabels, HashSet<String> catLexicon, HashSet<String> slotLexicon, HashSet<String> distLexicon, HashSet<String> posLexicon) {
		Feature result = new Feature();

		String topCat = record.get(0).toString();
		String leftCat = record.get(1).toString();
		String rightCat = record.get(2).toString();
		String leftLeftCat = record.get(3).toString();
		String leftRightCat = record.get(4).toString();
		String rightLeftCat = record.get(5).toString();
		String rightRightCat = record.get(6).toString();
		String topCatWord = record.get(7).toString();
		String leftCatWord = record.get(8).toString();
		String rightCatWord = record.get(9).toString();
		String leftLeftCatWord = record.get(10).toString();
		String leftRightCatWord = record.get(11).toString();
		String rightLeftCatWord = record.get(12).toString();
		String rightRightCatWord = record.get(13).toString();
		String topCatPos = record.get(14).toString();
		String leftCatPos = record.get(15).toString();
		String rightCatPos = record.get(16).toString();
		String leftLeftCatPos = record.get(17).toString();
		String leftRightCatPos = record.get(18).toString();
		String rightLeftCatPos = record.get(19).toString();
		String rightRightCatPos = record.get(20).toString();

		String valueString;

		if ( hardLabels ) {
			valueString = record.get(21).toString();
			result.value = Double.parseDouble(valueString);
		} else {
			valueString = record.get(22).toString();
			result.value = Math.tanh(Double.parseDouble(valueString) / sigmoidScaleFactor);
		}

		catLexicon.add(topCat);
		catLexicon.add(leftCat);
		catLexicon.add(rightCat);
		catLexicon.add(leftLeftCat);
		catLexicon.add(leftRightCat);
		catLexicon.add(rightLeftCat);
		catLexicon.add(rightRightCat);
		posLexicon.add(topCatPos);
		posLexicon.add(leftCatPos);
		posLexicon.add(rightCatPos);
		posLexicon.add(leftLeftCatPos);
		posLexicon.add(leftRightCatPos);
		posLexicon.add(rightLeftCatPos);
		posLexicon.add(rightRightCatPos);

		result.add(topCat);
		result.add(leftCat);
		result.add(rightCat);
		result.add(leftLeftCat);
		result.add(leftRightCat);
		result.add(rightLeftCat);
		result.add(rightRightCat);
		result.add(topCatWord);
		result.add(leftCatWord);
		result.add(rightCatWord);
		result.add(leftLeftCatWord);
		result.add(leftRightCatWord);
		result.add(rightLeftCatWord);
		result.add(rightRightCatWord);
		result.add(topCatPos);
		result.add(leftCatPos);
		result.add(rightCatPos);
		result.add(leftLeftCatPos);
		result.add(leftRightCatPos);
		result.add(rightLeftCatPos);
		result.add(rightRightCatPos);

		return result;
	}

	@Override
	public INDArray makeVector(NeuralNetwork<? extends NNType> depnn) {
		// head category slot dependent distance head_pos dependent_pos value count
		String topCat = this.get(0);
		String leftCat = this.get(1);
		String rightCat = this.get(2);
		String leftLeftCat = this.get(3);
		String leftRightCat = this.get(4);
		String rightLeftCat = this.get(5);
		String rightRightCat = this.get(6);
		String topCatWord = this.get(7);
		String leftCatWord = this.get(8);
		String rightCatWord = this.get(9);
		String leftLeftCatWord = this.get(10);
		String leftRightCatWord = this.get(11);
		String rightLeftCatWord = this.get(12);
		String rightRightCatWord = this.get(13);
		String topCatPos = this.get(14);
		String leftCatPos = this.get(15);
		String rightCatPos = this.get(16);
		String leftLeftCatPos = this.get(17);
		String leftRightCatPos = this.get(18);
		String rightLeftCatPos = this.get(19);
		String rightRightCatPos = this.get(20);

		INDArray topCatVector = depnn.catEmbeddings.getINDArray(topCat);
		INDArray leftCatVector = depnn.catEmbeddings.getINDArray(leftCat);
		INDArray rightCatVector = depnn.catEmbeddings.getINDArray(rightCat);
		INDArray leftLeftCatVector = depnn.catEmbeddings.getINDArray(leftLeftCat);
		INDArray leftRightCatVector = depnn.catEmbeddings.getINDArray(leftRightCat);
		INDArray rightLeftCatVector = depnn.catEmbeddings.getINDArray(rightLeftCat);
		INDArray rightRightCatVector = depnn.catEmbeddings.getINDArray(rightRightCat);
		INDArray topCatWordVector = depnn.getWordVector(topCatWord);
		INDArray leftCatWordVector = depnn.getWordVector(leftCatWord);
		INDArray rightCatWordVector = depnn.getWordVector(rightCatWord);
		INDArray leftLeftCatWordVector = depnn.getWordVector(leftLeftCatWord);
		INDArray leftRightCatWordVector = depnn.getWordVector(leftRightCatWord);
		INDArray rightLeftCatWordVector = depnn.getWordVector(rightLeftCatWord);
		INDArray rightRightCatWordVector = depnn.getWordVector(rightRightCatWord);
		INDArray topCatPosVector = depnn.posEmbeddings.getINDArray(topCatPos);
		INDArray leftCatPosVector = depnn.posEmbeddings.getINDArray(leftCatPos);
		INDArray rightCatPosVector = depnn.posEmbeddings.getINDArray(rightCatPos);
		INDArray leftLeftCatPosVector = depnn.posEmbeddings.getINDArray(leftLeftCatPos);
		INDArray leftRightCatPosVector = depnn.posEmbeddings.getINDArray(leftRightCatPos);
		INDArray rightLeftCatPosVector = depnn.posEmbeddings.getINDArray(rightLeftCatPos);
		INDArray rightRightCatPosVector = depnn.posEmbeddings.getINDArray(rightRightCatPos);

		return Nd4j.concat(1,
							topCatVector,
							leftCatVector,
							rightCatVector,
							leftLeftCatVector,
							leftRightCatVector,
							rightLeftCatVector,
							rightRightCatVector,
							topCatWordVector,
							leftCatWordVector,
							rightCatWordVector,
							leftLeftCatWordVector,
							leftRightCatWordVector,
							rightLeftCatWordVector,
							rightRightCatWordVector,
							topCatPosVector,
							leftCatPosVector,
							rightCatPosVector,
							leftLeftCatPosVector,
							leftRightCatPosVector,
							rightLeftCatPosVector,
							rightRightCatPosVector);
	}

	@Override
	public void updateEmbeddings(INDArray errors, int w2vLayerSize, Embeddings catEmbeddings, Embeddings slotEmbeddings, Embeddings distEmbeddings, Embeddings posEmbeddings) {
		catEmbeddings.addEmbedding(this.get(0), errors, 0 * w2vLayerSize);
		catEmbeddings.addEmbedding(this.get(1), errors, 1 * w2vLayerSize);
		catEmbeddings.addEmbedding(this.get(2), errors, 2 * w2vLayerSize);
		catEmbeddings.addEmbedding(this.get(3), errors, 3 * w2vLayerSize);
		catEmbeddings.addEmbedding(this.get(4), errors, 4 * w2vLayerSize);
		catEmbeddings.addEmbedding(this.get(5), errors, 5 * w2vLayerSize);
		catEmbeddings.addEmbedding(this.get(6), errors, 6 * w2vLayerSize);
		posEmbeddings.addEmbedding(this.get(14), errors , 14 * w2vLayerSize);
		posEmbeddings.addEmbedding(this.get(15), errors , 15 * w2vLayerSize);
		posEmbeddings.addEmbedding(this.get(16), errors , 16 * w2vLayerSize);
		posEmbeddings.addEmbedding(this.get(17), errors , 17 * w2vLayerSize);
		posEmbeddings.addEmbedding(this.get(18), errors , 18 * w2vLayerSize);
		posEmbeddings.addEmbedding(this.get(19), errors , 19 * w2vLayerSize);
		posEmbeddings.addEmbedding(this.get(20), errors , 20 * w2vLayerSize);
	}
}
