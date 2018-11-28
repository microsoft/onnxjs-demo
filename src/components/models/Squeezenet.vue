<template>
  <ImageModelUI
    modelName="squeezenet"
    :modelFilepath="modelFilepath"
    :imageSize="224"
    :preprocess="preprocess"
    :imageUrls="imageUrls"
  ></ImageModelUI>
</template>

<script lang="ts">
import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import ImageModelUI from '../common/ImageModelUI.vue';
import {Tensor} from 'onnxjs';
import {Vue, Component} from 'vue-property-decorator';
import { SQUEEZENET_IMAGE_URLS } from '../../data/sample-image-urls';

const MODEL_FILEPATH_PROD = `/squeezenetV1_8.onnx`;
const MODEL_FILEPATH_DEV = '/squeezenetV1_8.onnx';

@Component({
  components: {
    ImageModelUI
  }
})

export default class SqueezeNet extends Vue{
  modelFilepath: string;
  imageUrls: Array<{text: string, value: string}>;

  constructor() {
    super();
    this.modelFilepath = process.env.NODE_ENV === 'production' ? MODEL_FILEPATH_PROD : MODEL_FILEPATH_DEV;
    this.imageUrls = SQUEEZENET_IMAGE_URLS;

  }

  preprocess(ctx: CanvasRenderingContext2D): Tensor {
    const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
    const { data, width, height } = imageData;

    // data processing
    const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [1, 3, width, height]);

    ops.subseq(dataTensor.pick(2, null, null), 103.939);
    ops.subseq(dataTensor.pick(1, null, null), 116.779);
    ops.subseq(dataTensor.pick(0, null, null), 123.68);
    ops.assign(dataProcessedTensor.pick(0, 0, null, null), dataTensor.pick(null, null, 2));
    ops.assign(dataProcessedTensor.pick(0, 1, null, null), dataTensor.pick(null, null, 1));
    ops.assign(dataProcessedTensor.pick(0, 2, null, null), dataTensor.pick(null, null, 0));

    const tensor = new Tensor(new Float32Array(width * height * 3), 'float32', [1, 3, width, height]);
    (tensor.data as Float32Array).set(dataProcessedTensor.data);
    return tensor;
  }
}
</script>
