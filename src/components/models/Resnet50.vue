<template>
  <!-- <div class="demo"> -->
    <ImageModelUI
      :modelFilepath="modelFilepath"
      :imageSize="224"      
      :imageUrls="imageUrls"
      :preprocess="preprocess"
      :getPredictedClass="getPredictedClass"
    ></ImageModelUI>
  <!-- </div> -->
</template>

<script lang="ts">
import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import ImageModelUI from '../common/ImageModelUI.vue';
import {Tensor} from 'onnxjs';
import {Vue, Component} from 'vue-property-decorator';
import {RESNET50_IMAGE_URLS} from '../../data/sample-image-urls';
import {imagenetUtils} from '../../utils/index';

const MODEL_FILEPATH_PROD = `/resnet50_8.onnx`;
const MODEL_FILEPATH_DEV = '/resnet50_8.onnx';

@Component({
  components: {
    ImageModelUI
  }
})

export default class Resnet50 extends Vue{
  modelFilepath: string;
  imageUrls: Array<{text: string, value: string}>;
  constructor() {
    super();
    this.modelFilepath = process.env.NODE_ENV === 'production' ? MODEL_FILEPATH_PROD : MODEL_FILEPATH_DEV;
    this.imageUrls = RESNET50_IMAGE_URLS;
  }

  preprocess(ctx: CanvasRenderingContext2D): Tensor {
    const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
    const { data, width, height } = imageData;

    // data processing
    const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [1, 3, width, height]);

    ops.divseq(dataTensor, 128.0);
    ops.subseq(dataTensor, 1.0);

    ops.assign(dataProcessedTensor.pick(0, 0, null, null), dataTensor.pick(null, null, 2));
    ops.assign(dataProcessedTensor.pick(0, 1, null, null), dataTensor.pick(null, null, 1));
    ops.assign(dataProcessedTensor.pick(0, 2, null, null), dataTensor.pick(null, null, 0));

    const tensor = new Tensor(new Float32Array(3 * width * height), 'float32', [1, 3, width, height]);
    (tensor.data as Float32Array).set(dataProcessedTensor.data);
    return tensor;
  }

  getPredictedClass(output: Float32Array): {} {
    if (!output || output.length === 0) {
      const empty = [];
      for (let i = 0; i < 5; i++) {
        empty.push({ name: '-', probability: 0, index: 0 });
      }
      return empty;
    }
    return imagenetUtils.imagenetClassesTopK(output, 5);
  }
}
</script>
