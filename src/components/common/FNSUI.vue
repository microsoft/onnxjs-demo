<template>
  <div>
    <!-- session Loading and Initializing Indicator -->
    <model-status v-if="modelLoading || modelInitializing" 
      :modelLoading="modelLoading"
      :modelInitializing="modelInitializing"
    ></model-status>
    <v-container fluid>
      <!-- Utility bar to select session backend configs. -->
      <v-layout justify-center align-center style="margin: auto; width: 40%; padding: 30px">      
        <div class="select-backend"> Select Backend: </div>
        <v-select
          v-model="sessionBackend"
          :disabled="modelLoading || modelInitializing || sessionRunning"
          :items="backendSelectList"
          label="Switch Backend"
          :menu-props="{maxHeight:'750'}"
          solo single-line hide-details
        ></v-select>      
      </v-layout>
      <v-layout>
        <v-flex v-if="modelLoadingError" style="padding-bottom: 30px;" class="error-message">
          Error: Current backend is not supported on your machine. Try Selecting a different backend.
        </v-flex>
      </v-layout>

      <v-layout row wrap justify-space-around class="image-panel elevation-1">
        <!-- model status -->
        <div v-if="imageLoading || sessionRunning" class="loading-indicator">
        <v-progress-circular indeterminate color="primary" />
        </div>
        <!-- <v-layout column wrap> -->
        <!-- select input images -->
        <v-flex sm6 md4 align-center justify-start column fill-height>
          <v-flex sm3 md4 align-center justify-center style="margin: auto; padding-bottom: 30px" fill-width>
            <!-- <v-select
          v-model="styleSelect"
          :items="styleselectlist"
          label="Select style"
          persistent-hint
          return-object
          single-line
        ></v-select> -->
              <v-select v-model="styleSelect"
                :disabled="modelLoading || modelInitializing || modelLoadingError"
                :items="styleSelectList"
                label="Select style"
                :menu-props="{maxHeight:'750'}"
                solo single-line hide-details
              ></v-select>
              
            </v-flex>
          <v-layout align-center> 
            <v-flex sm4>
              <v-select
                v-model="imageURLSelect"
                :disabled="modelLoading || modelInitializing || modelLoadingError"
                :items="imageURLSelectList"
                label="Select image"
                :menu-props="{maxHeight:'750'}"
                solo single-line hide-details
              ></v-select>
              
            </v-flex>
            <v-flex class="text-xs-center">or</v-flex>
            <label :disabled="modelLoading || modelInitializing || modelLoadingError" class="inputs">
              <div>
                <span>UPLOAD IMAGE</span>
              </div>
              <input style="display: none" type="file" @change="handleFileChange"/>
            </label>
          </v-layout>
          <!-- input image -->
          <div v-if="imageLoadingError" class="error-message" style="padding-top: 30px">Error loading URL</div>
          <v-flex align-center justify-space-between class="canvas-container">
            <canvas id="input-canvas"
                :width="imageSize"
                :height="imageSize"
            ></canvas>
          </v-flex>
        </v-flex>
        <!-- </v-layout> -->
        <v-flex sm6 md4 column fill-height class="output-container" style="padding-top: 90px">
          <v-flex class="inference-time-class">
            <span class="inference-time">Inference Time: </span>
            <span v-if="inferenceTime > 0" class="inference-time-value">{{ inferenceTime.toFixed(1) }} ms </span>
            <span v-else>-</span>
          </v-flex>
          <canvas id="output-canvas"
                :width="imageSize"
                :height="imageSize"
            ></canvas>
        </v-flex>
      </v-layout>
      
    </v-container>
       
  </div>
</template>

<script lang="ts">
import loadImage from 'blueimp-load-image';
import {runModelUtils} from '../../utils';

import modelStatus from './ModelStatus.vue';
import {InferenceSession, Tensor} from 'onnxjs';
import {Vue, Component, Prop, Watch} from 'vue-property-decorator';

@Component({
  components: {
    modelStatus
  }
})

export default class FNSUI extends Vue{
  @Prop({ type: String, required: true }) modelFilepath!: string;
  @Prop({ type: Number, required: true }) imageSize!: number;  
  @Prop({ type: Array, required: true}) imageUrls!: Array<{text: string, value: string}>;
  @Prop({ type: Array, required: true}) styles!: Array<{text: string, value: string}>;
  @Prop({ type: Function, required: true }) preprocess!: (ctx: CanvasRenderingContext2D) => Tensor;
  @Prop({ type: Function, required: true }) postprocess!: (ten: Tensor, ctx: CanvasRenderingContext2D) => void;
  //@Prop({ type: Function, required: true }) getPredictedClass !: (output: Float32Array) => {};

  sessionBackend: string;
  backendSelectList: Array<{text: string, value: string}>;
  modelLoading: boolean;
  modelInitializing: boolean;
  modelLoadingError: boolean;
  sessionRunning: boolean;  
  session: InferenceSession | undefined;
  gpuSession: { [style: string]: InferenceSession };
  cpuSession: { [style: string]: InferenceSession };

  styleSelect: string;
  styleSelectList: Array<{text: string, value: string}>;
  // RETHINK THIS************

  inferenceTime: number;
  imageURLInput: string;
  imageURLSelect: null;
  imageURLSelectList: Array<{text: string, value: string}>;
  imageLoading: boolean;
  imageLoadingError: boolean;
  output: Tensor.DataType;
  modelFile: ArrayBuffer;

  constructor() {
    super();
    this.sessionBackend = 'webgl';
    this.backendSelectList = [{text: 'GPU-WebGL', value: 'webgl'}, {text: 'CPU-WebAssembly', value: 'wasm'}];
    this.modelLoading = true;
    this.modelInitializing = true;
    this.modelLoadingError = false;
    this.sessionRunning = false;
    this.inferenceTime = 0;
    this.imageURLInput = '';
    this.imageURLSelect = null;
    this.imageURLSelectList = this.imageUrls;
    this.styleSelect = this.styles[0].value;
    this.styleSelectList = this.styles;
    this.imageLoading = false;
    this.imageLoadingError = false;
    this.output = [];
    this.modelFile = new ArrayBuffer(0);
    this.gpuSession = {};
    this.cpuSession = {};
  }

  async created() {
    // fetch the model file to be used later
    const response = await fetch(this.modelFilepath);
    this.modelFile = await response.arrayBuffer();
    console.log(this.modelFilepath);
    try {
      await this.initSession();
    } catch (e) {
      this.sessionBackend = 'wasm';
    }
  }

  async initSession() {
    this.sessionRunning = false;
    this.modelLoadingError = false;
    /** BACKEND */
    if (this.sessionBackend === 'webgl') { 
      if (this.gpuSession[this.styleSelect]) {
        this.session = this.gpuSession[this.styleSelect];
        return;
      }
      this.modelLoading = true;
      this.modelInitializing = true;  
      this.gpuSession[this.styleSelect] = new InferenceSession({backendHint: this.sessionBackend});
      this.session = this.gpuSession[this.styleSelect];
    }
    if (this.sessionBackend === 'wasm') {        
      if (this.cpuSession[this.styleSelect]) {
        this.session = this.cpuSession[this.styleSelect];
        return;
      }
      this.modelLoading = true;
      this.modelInitializing = true;  
      this.cpuSession[this.styleSelect] = new InferenceSession({backendHint: this.sessionBackend});
      this.session = this.cpuSession[this.styleSelect];
    }    
    console.log('before loading model');
    console.log(this.modelFile);

    console.log('SESSION ' + this.sessionBackend);

    try {
      await this.session!.loadModel(this.modelFile);
      // console.log('after loading model');
    } catch (e) {
      this.modelLoading = false;
      this.modelInitializing = false;
      console.log(e);
      console.log(this.styleSelectList[0].text);
      if (this.sessionBackend === 'webgl') {
        this.gpuSession = {};
      } else {
        this.cpuSession = {};
      }
      throw new Error('Error: Backend not supported. ');
    }
    this.modelLoading = false;
    // warm up session with a sample tensor. Use setTimeout(..., 0) to make it an async execution so 
    // that UI update can be done.
    if (this.sessionBackend === 'webgl') {
      setTimeout(() => {
        runModelUtils.warmupModel(this.session!, [1, 3, this.imageSize, this.imageSize]);
        this.modelInitializing = false;

      }, 0);
    } else {
      await runModelUtils.warmupModel(this.session!, [1, 3, this.imageSize, this.imageSize]);
      this.modelInitializing = false;
    }
  }

  @Watch('sessionBackend')
  async onSessionBackendChange(newVal: string) {
    this.sessionBackend = newVal;
    this.clearAll();
    try {
      await this.initSession();
    } catch (e) {
      console.log(e);
      // CHANGE TO TRUE LATER
      this.modelLoadingError = true;
    }
    return newVal;
  }

  @Watch('imageURLSelect')
  onImageURLSelectChange(newVal: string) {
    this.imageURLInput = newVal;
    this.loadImageToCanvas(newVal);
  }

  @Watch('styleSelect')
  async styleSelectChange(newStyle: string) {
    this.styleSelect = newStyle;
    const response = await fetch(this.styleSelect);
    this.modelFile = await response.arrayBuffer();
    this.clearSome();
    try {
      await this.initSession();
    } catch (e) {
      this.modelLoadingError = true;
    }
    if (!this.imageLoadingError && !this.modelLoadingError && this.imageURLInput !== '') {
      this.sessionRunning = true;
      this.$nextTick(function() {
        setTimeout(() => {
          this.runModel();
        }, 10);
      });
    }
    return newStyle;
  }
  
  beforeDestroy() {
    this.session = undefined;
    this.gpuSession = {};
    this.cpuSession = {};
  }

  // get outputClasses() {
  //   return this.getPredictedClass(Array.prototype.slice.call(this.output));
  // }

  onImageURLInputEnter(e: any) {
      this.imageURLSelect = null;
      this.loadImageToCanvas(e.target.value);
  }

  handleFileChange(e: any) {
    this.$emit('input', e.target.files[0]);
    this.loadImageToCanvas(e.target.files[0]);
  }

  loadImageToCanvas(url: string) {
    if (!url) {
        this.clearAll();
        return;
    }
    this.imageLoading = true;
    loadImage(
        url,
        img => {
        if ((img as Event).type === 'error') {
            this.imageLoadingError = true;
            this.imageLoading = false;
        } else {
            // load image data onto input canvas
            const element = document.getElementById('input-canvas') as HTMLCanvasElement;
            if (element) {
              const ctx = element.getContext('2d');
              if (ctx) {
                ctx.drawImage(img as HTMLImageElement, 0, 0);
                this.imageLoadingError = false;
                this.imageLoading = false;
                this.sessionRunning = true;
                this.output = [];
                this.inferenceTime = 0;
                // session predict
                this.$nextTick(function() {
                setTimeout(() => {
                    this.runModel();
                }, 10);
                });
              }
            }
        }
        },
        {
        maxWidth: this.imageSize,
        maxHeight: this.imageSize,
        cover: true,
        crop: true,
        canvas: true,
        crossOrigin: 'Anonymous',
        }
    );
  }

  async runModel() {
    const element = document.getElementById('input-canvas') as HTMLCanvasElement;
    const ctx = element.getContext('2d') as CanvasRenderingContext2D;    
    const outElement = document.getElementById('output-canvas') as HTMLCanvasElement;
    const outCtx = outElement.getContext('2d') as CanvasRenderingContext2D;

    const preprocessedData = this.preprocess(ctx);
    let tensorOutput = null;
    [tensorOutput, this.inferenceTime] = await runModelUtils.runModel(this.session!, preprocessedData);
    // this.output = tensorOutput.data;

    this.postprocess(tensorOutput, outCtx);
    this.sessionRunning = false;
  }

  clearAll() {
    this.sessionRunning = false;
    this.inferenceTime = 0;
    this.imageURLInput = '';
    this.imageURLSelect = null;
    this.imageLoading = false;
    this.imageLoadingError = false;
    this.output = [];
    
    const element = document.getElementById('input-canvas') as HTMLCanvasElement;
    if (element) {
      const ctx = element.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      }
    }
  }

  clearSome() {
    this.sessionRunning = false;
    this.inferenceTime = 0;
    this.imageLoading = false;
    this.imageLoadingError = false;
    this.output = [];
  }
}
</script>

<style lang="postcss" scoped>
@import '../../variables.css';
.image-panel {
  padding: 80px 0px 80px 0px;
  margin: auto;
  background-color: white;
  position: relative;
  width: 85%;
  height: 100%;
  & .loading-indicator {
    position: absolute;
    top: 5px;
    left: 5px;
  }

}

.inputs {
  margin: auto;
  background: #f5f5f5;
  box-shadow: 0 3px 1px -2px rgba(0, 0, 0, .2),0 2px 2px 0 rgba(0, 0, 0, .14),0 1px 5px 0 rgba(0, 0, 0, .12);
  align-items: center;
  border-radius: 2px;
  display: inline-flex;
  height: 40px;
  font-size: 14px;
  transition: .3s cubic-bezier(.25,.8,.5,1),color 1ms;
  padding: 0 16px;
}

/* .inputs:focus, .inputs:hover {
	position: relative;
  background: rgba(0, 0, 0, .12);
}

.input-label {
  font-family: var(--font-sans-serif);
  font-size: 16px;
  color: var(--color-blue);
  text-align: left;
  user-select: none;
  cursor: default;
} */

.canvas-container {
  position: relative;
  text-align: center;
  & #input-canvas {
    background: #eeeeee;
    margin-top: 40px;
  }

}

.output-container {
  position: relative;
  text-align: center;
  & #output-canvas {
    background: #eeeeee;
    margin-top: 49px;
  }
}
</style>
