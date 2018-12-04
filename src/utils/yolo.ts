// Heavily derived from YAD2K (https://github.com/allanzelener/YAD2K)
import classNames from '../data/yolo_classes';
import {yolo} from './index';

// export const YOLO_ANCHORS = tf.tensor2d([
//   [0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434],
//   [7.88282, 3.52778], [9.77052, 9.16828],
// ]);

export const YOLO_ANCHORS = tf.tensor2d([[1.08, 1.19], [3.42, 4.41], 
  [6.63, 11.38], [9.42, 5.11], 
  [16.62, 10.52]
]);

const DEFAULT_FILTER_BOXES_THRESHOLD = 0.01;
const DEFAULT_IOU_THRESHOLD = 0.4;
const DEFAULT_CLASS_PROB_THRESHOLD = 0.3;
const INPUT_DIM = 416;

export async function postprocess(outputTensor: tf.Tensor, numClasses: number) {
  const [boxXy, boxWh, boxConfidence, boxClassProbs ] = yolo_head(outputTensor, YOLO_ANCHORS, 20);
    console.log('time4 = ' + new Date().getTime());
		const allBoxes = yolo_boxes_to_corners(boxXy, boxWh);
    console.log('time5 = ' + new Date().getTime());
		const [outputBoxes, scores, classes] = await yolo_filter_boxes(
    allBoxes, boxConfidence, boxClassProbs, DEFAULT_FILTER_BOXES_THRESHOLD);
    // If all boxes have been filtered out
    console.log('time6 = ' + new Date().getTime());
		if (outputBoxes == null) {
      return [];
    }

  const width = tf.scalar(INPUT_DIM);
  const height = tf.scalar(INPUT_DIM);

  const imageDims = tf.stack([height, width, height, width]).reshape([1,4]);

  const boxes: tf.Tensor<tf.Rank> = tf.mul(outputBoxes, imageDims);

  const [ preKeepBoxesArr, scoresArr ] = await Promise.all([
    boxes.data(), scores.data(),
  ]);

  const [ keepIndx, boxesArr, keepScores ] = non_max_suppression(
    preKeepBoxesArr,
    scoresArr,
    DEFAULT_IOU_THRESHOLD,
  );

  const classesIndxArr = await classes.gather(tf.tensor1d(keepIndx, 'int32')).data() as Float32Array;

  const results: any[] = [];

  classesIndxArr.forEach((classIndx, i) => {
    const classProb = keepScores[i];
    if (classProb < DEFAULT_CLASS_PROB_THRESHOLD) {
      return;
    }

    const className = classNames[classIndx];
    let [top, left, bottom, right] = boxesArr[i];

    top = Math.max(0, top);
    left = Math.max(0, left);
    bottom = Math.min(416, bottom);
    right = Math.min(416, right);

    const resultObj = {
      className,
      classProb,
      bottom,
      top,
      left,
      right,
    };

    results.push(resultObj);
  });
  console.log(results);
  return results;
}

export async function yolo_filter_boxes(
    boxes: tf.Tensor,
    boxConfidence: tf.Tensor,
    boxClassProbs: tf.Tensor,
    threshold: number
  ) {
    const boxScores = tf.mul(boxConfidence, boxClassProbs);
    const boxClasses = tf.argMax(boxScores, -1);
    const boxClassScores = tf.max(boxScores, -1);
    // Many thanks to @jacobgil
    // Source: https://github.com/ModelDepot/tfjs-yolo-tiny/issues/6#issuecomment-387614801
    const predictionMask = tf.greaterEqual(boxClassScores, tf.scalar(threshold)).as1D();
  
    const N = predictionMask.size;
    // linspace start/stop is inclusive.
    const allIndices = tf.linspace(0, N - 1, N).toInt();
    const negIndices = tf.zeros([N], 'int32');
    const indices = tf.where(predictionMask, allIndices, negIndices) as tf.Tensor<Rank.R1>;
  
    return [
      tf.gather(boxes.reshape([N, 4]), indices),
      tf.gather(boxClassScores.flatten(), indices),
      tf.gather(boxClasses.flatten(), indices),
    ];
  }

/**
 * Given XY and WH tensor outputs of yolo_head, returns corner coordinates.
 * @param {tf.Tensor} box_xy Bounding box center XY coordinate Tensor
 * @param {tf.Tensor} box_wh Bounding box WH Tensor
 * @returns {tf.Tensor} Bounding box corner Tensor
 */
export function yolo_boxes_to_corners(boxXy: tf.Tensor, boxWh: tf.Tensor) {
  const two = tf.tensor1d([2.0]);
  const boxMins = tf.sub(boxXy, tf.div(boxWh, two));
  const boxMaxes = tf.add(boxXy, tf.div(boxWh, two));

  const dim0 = boxMins.shape[0];
  const dim1 = boxMins.shape[1];
  const dim2 = boxMins.shape[2];
  const size = [dim0, dim1, dim2, 1];

  return tf.concat([
    boxMins.slice([0, 0, 0, 1], size),
    boxMins.slice([0, 0, 0, 0], size),
    boxMaxes.slice([0, 0, 0, 1], size),
    boxMaxes.slice([0, 0, 0, 0], size),
  ], 3);
}

/**
 * Filters/deduplicates overlapping boxes predicted by YOLO. These
 * operations are done on CPU as AFAIK, there is no tfjs way to do it
 * on GPU yet.
 * @param {TypedArray} boxes Bounding box corner data buffer from Tensor
 * @param {TypedArray} scores Box scores data buffer from Tensor
 * @param {Number} iouThreshold IoU cutoff to filter overlapping boxes
 */
export function non_max_suppression(boxes: Float32Array | Int32Array | Uint8Array, 
	scores: Float32Array | Int32Array | Uint8Array, iouThreshold: number) {
  // Zip together scores, box corners, and index
  const zipped = [];
  for (let i=0; i<scores.length; i++) {
    zipped.push([
      scores[i], [boxes[4*i], boxes[4*i+1], boxes[4*i+2], boxes[4*i+3]], i,
    ]);
  }
  // Sort by descending order of scores (first index of zipped array)
  const sortedBoxes = zipped.sort((a: number[], b: number[]) => b[0] - a[0]);

  const selectedBoxes: any[] = [];

  // Greedily go through boxes in descending score order and only
  // return boxes that are below the IoU threshold.
  sortedBoxes.forEach((box: any[]) => {
    let add = true;
    for (let i=0; i < selectedBoxes.length; i++) {
      // Compare IoU of zipped[1], since that is the box coordinates arr
      // TODO: I think there's a bug in this calculation
      const curIou = box_iou(box[1], selectedBoxes[i][1]);
      if (curIou > iouThreshold) {
        add = false;
        break;
      }
    }
    if (add) {
      selectedBoxes.push(box);
    }
  });

  // Return the kept indices and bounding boxes
  return [
    selectedBoxes.map(e => e[2]),
    selectedBoxes.map(e => e[1]),
    selectedBoxes.map(e => e[0]),
  ];
}

// Convert yolo output to bounding box + prob tensors
export function yolo_head(feats: tf.Tensor, anchors: tf.Tensor, numClasses: number) {
  const numAnchors = anchors.shape[0];

  const anchorsArray = tf.reshape(anchors, [1, 1, numAnchors, 2]);

  const convDims = feats.shape.slice(1, 3);

  // For later use
  const convDims0 = convDims[0];
  const convDims1 = convDims[1];

  let convHeightIndex = tf.range(0, convDims[0]);
  let convWidthIndex = tf.range(0, convDims[1]);

  convHeightIndex = tf.tile(convHeightIndex, [convDims[1]]);

  convWidthIndex = tf.tile(tf.expandDims(convWidthIndex, 0), [convDims[0], 1]);
  convWidthIndex = tf.transpose(convWidthIndex).flatten();

  let convIndex = tf.transpose(tf.stack([convHeightIndex, convWidthIndex]));
  convIndex = tf.reshape(convIndex, [convDims[0], convDims[1], 1, 2]);
  convIndex = tf.cast(convIndex, feats.dtype);

  feats = tf.reshape(feats, [convDims[0], convDims[1], numAnchors, numClasses + 5]);
  const convDimsTensor = tf.cast(tf.reshape(tf.tensor1d(convDims), [1,1,1,2]), feats.dtype);

  let boxXy = tf.sigmoid(feats.slice([0,0,0,0], [convDims0, convDims1, numAnchors, 2]));
  let boxWh = tf.exp(feats.slice([0,0,0, 2], [convDims0, convDims1, numAnchors, 2]));
  const boxConfidence = tf.sigmoid(feats.slice([0, 0, 0, 4], [convDims0, convDims1, numAnchors, 1]));
  const boxClassProbs = tf.softmax(feats.slice([0, 0, 0, 5],[convDims0, convDims1, numAnchors, numClasses]));

  boxXy = tf.div(tf.add(boxXy, convIndex), convDimsTensor);
	boxWh = tf.div(tf.mul(boxWh, anchorsArray), convDimsTensor);
  // boxXy = tf.mul(tf.add(boxXy, convIndex), 32);
  // boxWh = tf.mul(tf.mul(boxWh, anchorsArray), 32);
  return [ boxXy, boxWh, boxConfidence, boxClassProbs ];
}

export function box_intersection(a: number[], b: number[]) {
	const w = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
	const h = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
	if (w < 0 || h < 0) {
		return 0;
	}
	return w * h;
}

export function box_union(a: number[], b: number[]) {
	const i = box_intersection(a, b);
	return (a[3] - a[1]) * (a[2] - a[0]) + (b[3] - b[1]) * (b[2] - b[0]) - i;
}

export function box_iou(a: number[], b: number[]) {
	return box_intersection(a, b) / box_union(a, b);
}