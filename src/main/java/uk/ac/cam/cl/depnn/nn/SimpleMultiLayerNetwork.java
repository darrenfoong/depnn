package uk.ac.cam.cl.depnn.nn;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class SimpleMultiLayerNetwork {
	private int INPUT_LAYER_SIZE;
	private int HIDDEN_LAYER_SIZE;
	private int OUTPUT_LAYER_SIZE;

	private INDArray w_h;
	private INDArray w_out;
	private INDArray b_h;
	private INDArray b_out;

	private final static Logger logger = LogManager.getLogger(SimpleMultiLayerNetwork.class);

	public SimpleMultiLayerNetwork(String coefficientsFile, int inputLayerSize, int hiddenLayerSize, int outputLayerSize) throws IOException {
		INPUT_LAYER_SIZE = inputLayerSize;
		HIDDEN_LAYER_SIZE = hiddenLayerSize;
		OUTPUT_LAYER_SIZE = outputLayerSize;

		DataInputStream dis = new DataInputStream(new FileInputStream(coefficientsFile));
		INDArray newParams = Nd4j.read(dis);

		int idx = 0;
		int range = 0;

		range = INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE;
		w_h = newParams.get(NDArrayIndex.point(0), NDArrayIndex.interval(idx, range + idx));
		w_h = w_h.reshape(HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE).transposei();
		idx += range;

		range = 1 * HIDDEN_LAYER_SIZE;
		b_h = newParams.get(NDArrayIndex.point(0), NDArrayIndex.interval(idx, range + idx));
		b_h = b_h.reshape(HIDDEN_LAYER_SIZE, 1).transposei();
		idx += range;

		range = HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE;
		w_out = newParams.get(NDArrayIndex.point(0), NDArrayIndex.interval(idx, range + idx));
		w_out = w_out.reshape(OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE).transposei();
		idx += range;

		range = 1 * OUTPUT_LAYER_SIZE;
		b_out = newParams.get(NDArrayIndex.point(0), NDArrayIndex.interval(idx, range + idx));
		b_out = b_out.reshape(OUTPUT_LAYER_SIZE, 1).transposei();
		idx += range;
	}

	public INDArray relui(INDArray inputs) {
		for ( int i = 0; i < inputs.rows(); i++ ) {
			for ( int j = 0; j < inputs.columns(); j++ ) {
				if ( inputs.getDouble(i, j) < 0 ) {
					inputs.put(i, j, 0);
				}
			}
		}

		return inputs;
	}

	public INDArray softmaxi(INDArray inputs) {
		for ( int i = 0; i < inputs.rows(); i++ ) {
			double sum = 0.0;

			for ( int j = 0; j < inputs.columns(); j++ ) {
				sum += Math.exp(inputs.getDouble(i, j));
			}

			for ( int j = 0; j < inputs.columns(); j++ ) {
				inputs.put(i, j, Math.exp(inputs.getDouble(i, j)) / sum);
			}
		}

		return inputs;
	}

	public INDArray output(INDArray inputs, boolean training) {
		INDArray hidden_layer = relui(inputs.mmul(w_h).addiRowVector(b_h));
		return softmaxi(hidden_layer.mmul(w_out).addiRowVector(b_out));
	}
}