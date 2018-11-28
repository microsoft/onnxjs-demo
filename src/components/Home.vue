<template>
  <div class="demo home text-xs-center">
    <v-img class="white--text" :src="require('@/assets/background.png')" height="600px">
      <v-container padding-top="16px">
        <v-layout column justify-center align-center>
          <v-flex class="onnx">ONNX.JS</v-flex>
          <v-flex class="run-onnx">Run ONNX model in the browser</v-flex>
          <v-flex class="onnx-info">Interactive ML without install and device independent<br>
  Latency of server-client communication reduced<br>
  Privacy and security ensured<br>
  GPU acceleration</v-flex>
        </v-layout>
      </v-container>
    </v-img>
    <div v-for="info in demoInfo" :key="info.path" style="width:100%; margin-top:30px">
      <router-link :to="`/${info.path}`">
        <div class="demo-card">
          <div class="demo-card-image"><img :src="info.imagePath"/></div>
          <div class="demo-card-heading">{{ info.title }}</div>
        </div>
      </router-link>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
const DEMO_INFO = [
  { title: 'ResNet50, trained on ImageNet', path: 'resnet50', imagePath: require('@/assets/resnet50.png') },
  { title: 'SqueezeNet, trained on ImageNet', path: 'squeezenet', imagePath: require('@/assets/squeezenet.png') },
  { title: 'Emotion FerPlus', path: 'emotion', imagePath: require('@/assets/emotion.png') },
  { title: 'MNIST', path: 'mnist', imagePath: require('@/assets/mnist.png') },
];

@Component
export default class HomePage extends Vue {
  demoInfo: Array<{title: string, path: string, imagePath: string}> = DEMO_INFO;

  constructor() {
    super();
    this.demoInfo = DEMO_INFO;
  }
}
</script>

<style scoped lang="postcss">
@import '../variables.css';

.home {
  background: var(--color-blue);
  height: 100%;
  width: 100%;
}

.demo-card {
  font-family: var(--font-sans-serif);
  width: 100%;
  max-width: 1000px;
  height: 90px;
  margin: 30px auto;
  background: white;
  border: 1px solid whitesmoke;
  cursor: default;
  user-select: none;
  display: inline-flex;
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
  overflow: hidden;
  box-shadow: 3px 3px #062D5B;
  transition: box-shadow 0.2s ease-out;

  &:first-child {
    margin-top: 0;
  }

  &:hover {
    box-shadow: 3px 3px 5px var(--color-blue-light);
    cursor: pointer;

    & .demo-card-heading {
      color: var(--color-blue);
    }
  }
}

.demo-card-heading {
  flex: 1;
  padding: 20px;
  text-align: center;
  color: var(--color-lightgray);
  font-size: 18px;
  transition: color 0.2s ease-out;
}

.demo-card-image {
  height: 90px;

  & img {
    width: auto;
    height: 100%;
  }
}

.onnx {
  margin-top: 100px;
  font-size: 50px;
}

.run-onnx {
  font-size: 30px;
}

.onnx-info {
  font-family: var(--font-sans-serif-regular);
  font-size: 18px;
  width: 600px;
  margin-top: 80px;
}
</style>

