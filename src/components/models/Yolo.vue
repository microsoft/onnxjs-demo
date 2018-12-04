<template>
  <WebcamModel
    modelName="Yolo"
    :hasWebGL="hasWebGL"
    :modelFilepath="modelFilepath"
    :imageSize="416"
    :imageUrls="imageUrls"
    :run="runModel"
    :warmupModel="warmupModel"
  ></WebcamModel>
</template>

<script lang="ts">
import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import WebcamModel from '../common/WebcamModelUI.vue';
import {Vue, Component, Prop} from 'vue-property-decorator';
import {runModelUtils, yolo} from '../../utils/index';
import { YOLO_IMAGE_URLS } from '../../data/sample-image-urls';
// import * as tf from '@tensorflow/tfjs';
import {Tensor, InferenceSession} from 'onnxjs';

const MODEL_FILEPATH_PROD = `/yolo.onnx`;
const MODEL_FILEPATH_DEV = '/yolo.onnx';

@Component({
  components: {
    WebcamModel
  }
})

export default class Yolo extends Vue{
  @Prop(Boolean) hasWebGL!: boolean;
  imageUrls: Array<{text: string, value: string}>;
  modelFilepath: string;

  constructor() {
    super();
    this.imageUrls = YOLO_IMAGE_URLS;
    this.modelFilepath = process.env.NODE_ENV === 'production' ? MODEL_FILEPATH_PROD : MODEL_FILEPATH_DEV;
  }

  warmupModel(session: InferenceSession) {
		return runModelUtils.warmupModel(session, [1, 3, 416, 416]);
  }
  
  preprocess(ctx: CanvasRenderingContext2D): Tensor {
    const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
    const { data, width, height } = imageData;
    // data processing
    const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [1, 3, width, height]);

    ops.assign(dataProcessedTensor.pick(0, 0, null, null), dataTensor.pick(null, null, 2));
    ops.assign(dataProcessedTensor.pick(0, 1, null, null), dataTensor.pick(null, null, 1));
    ops.assign(dataProcessedTensor.pick(0, 2, null, null), dataTensor.pick(null, null, 0));

    const tensor = new Tensor(new Float32Array(width* height* 3), 'float32', [1, 3, width, height]);
    (tensor.data as Float32Array).set(dataProcessedTensor.data);
    return tensor;
  }

  async runModel(ctx: CanvasRenderingContext2D, session: InferenceSession) {
    const preprocessedData = this.preprocess(ctx);
    console.log('preprocessedData ', preprocessedData);
    console.log('run model');
    console.log('time1 = ' + new Date().getTime());

    // const url = 'http://138.91.127.119/score';
    const [tensorData, inferenceTime] = await runModelUtils.runModel(session, preprocessedData);
    console.log(tensorData);
    try {
      const originalOutput = new Tensor(tensorData.data as Float32Array, 'float32', [1, 125, 13, 13]);
      console.log('originalOutput ', originalOutput);
      const outputTensor = yolo.transpose(originalOutput, [0, 2, 3, 1]);

      console.log('time3 = ' + new Date().getTime());
          
      // postprocessing
      const boxes = await yolo.postprocess(outputTensor, 20);
      console.log('postProcessedData ', boxes);
      boxes.forEach(box => {
        const {
          top, left, bottom, right, classProb, className,
        } = box;

        this.drawRect(left, top, right-left, bottom-top,
          `${className} Confidence: ${Math.round(classProb * 100)}% Time: ${inferenceTime.toFixed(1)}ms`);
      });
    } catch (e) {
      alert('Model is not valid!');
    }
  }
  
  drawRect(x: number, y: number, w: number, h: number, text = '', color = 'red') {
    const rect = document.createElement('div');
    rect.style.cssText = `top: ${y}px; left: ${x}px; width: ${w}px; height: ${h}px; border-color: ${color};`;
    const label = document.createElement('div');
    label.innerText = text;
    rect.appendChild(label);

    (document.getElementById('webcam-container') as HTMLElement).appendChild(rect);
  }

}
</script>