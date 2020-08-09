<template>
  <FNSUI
    :modelFilepath="modelFilepath"
    :imageSize="224"
    :imageUrls="imageUrls"
    :styles="styles"
    :preprocess="preprocess"
    :postprocess="postprocess"
  ></FNSUI>
</template>

<script lang="ts">
import ndarray from 'ndarray';
import ops from 'ndarray-ops';
import FNSUI from '../common/FNSUI.vue';
import {Tensor} from 'onnxjs';
import {Vue, Component} from 'vue-property-decorator';
import {SQUEEZENET_IMAGE_URLS} from '../../data/sample-image-urls';
import {FNS_STYLES} from '../../data/fns_styles';
//import {imagenetUtils, mathUtils} from '../../utils/index';

const MODEL_FILEPATH_PROD = `/onnxjs-demo/fns-models/mosaic-8.onnx`;
const MODEL_FILEPATH_DEV = '/fns-models/mosaic-8.onnx';

@Component({
  components: {
    FNSUI
  }
})

export default class FastNeuralStyle extends Vue{
  modelFilepath: string;
  imageUrls: Array<{text: string, value: string}>;
  styles: Array<{text: string, value: string}>;

  constructor() {
    super();
    this.modelFilepath = process.env.NODE_ENV === 'production' ? MODEL_FILEPATH_PROD : MODEL_FILEPATH_DEV;
    this.imageUrls = SQUEEZENET_IMAGE_URLS;
    this.styles = FNS_STYLES;
  }

  preprocess(ctx: CanvasRenderingContext2D): Tensor {
    const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
    const { data, width, height } = imageData;
    console.log('WIDTH ' + width);

    // data processing
    const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [1, 3, width, height]);
    //console.log('WIDTH ' + width);

    ops.assign(dataProcessedTensor.pick(0, 0, null, null), dataTensor.pick(null, null, 0));
    ops.assign(dataProcessedTensor.pick(0, 1, null, null), dataTensor.pick(null, null, 1));
    ops.assign(dataProcessedTensor.pick(0, 2, null, null), dataTensor.pick(null, null, 2));

    const tensor = new Tensor(new Float32Array(width * height * 3), 'float32', [1, 3, width, height]);
    (tensor.data as Float32Array).set(dataProcessedTensor.data); 
    return tensor;
  }
  
  postprocess(ten: Tensor, ctx: CanvasRenderingContext2D): void {
    // clip - Uint8ClampedArray
    const data = new Uint8ClampedArray(ten.data as Float32Array);

    //transpose
    const width = ten.dims[2];
    const height = ten.dims[3];

    const dataTensor = ndarray(data, [1, 3, width, height]);
    const dataProcessedTensor = ndarray(new Uint8ClampedArray(width * height * 4), [width, height, 4]);

    //model gives a 4D tensor - transpose it back by picking and assigning to 3D tensor
    ops.assign(dataProcessedTensor.pick(null, null, 0), dataTensor.pick(0, 0, null, null));
    ops.assign(dataProcessedTensor.pick(null, null, 1), dataTensor.pick(0, 1, null, null));
    ops.assign(dataProcessedTensor.pick(null, null, 2), dataTensor.pick(0, 2, null, null));

    for (let i = 3; i < dataProcessedTensor.size; i += 4) {
      dataProcessedTensor.data[i] = 255;
    }

    // const imageData = new ImageData(dataProcessedTensor.data as Uint8ClampedArray, width, height);
    const imageData = ctx.createImageData(width, height);
    imageData.data.set(dataProcessedTensor.data);
    console.log(imageData);
    ctx.putImageData(imageData, 0, 0);
  }

  // getPredictedClass(res: Float32Array): {} {
  //   if (!res || res.length === 0) {
  //     const empty = [];
  //     for (let i = 0; i < 5; i++) {
  //       empty.push({ name: '-', probability: 0, index: 0 });
  //     }
  //     return empty;
  //   }
  //   const output = mathUtils.softmax(Array.prototype.slice.call(res));
  //   return imagenetUtils.imagenetClassesTopK(output, 5);
  // }
}
</script>
