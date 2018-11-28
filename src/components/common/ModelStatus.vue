<template>
  <v-layout justify-start align-center column class="model-status-background">
    <div class="model-status">{{message}}  </div>
    <v-flex>
      <v-progress-circular v-show="modelLoading | modelInitializing" indeterminate color="primary" height="250px"/>
    </v-flex>
  </v-layout>
</template>

<script lang="ts">
import {Vue, Component, Prop} from 'vue-property-decorator';

@Component
export default class ModelStatus extends Vue {
  @Prop({ type: Boolean, required: true }) modelLoading !: boolean;
  @Prop({ type: Boolean, required: true }) modelInitializing !: boolean;

  value: number;
  constructor() {
    super();
    this.value = 0;
  }

  get message() {
    if (this.modelLoading) {
      return 'Loading model...';
    } else if (this.modelInitializing) {
      return 'Loading model done. Initializing model...';
    } else {
      return '';
    }
  }

}
</script>

<style scoped lang="postcss">
@import '../../variables.css';

.model-status-background {
  position: absolute;
  z-index: 2;
  width: 100%;
  height: 100%;
  background-color: whitesmoke;
  opacity: 1;
  justify-content:center;
  text-align: center;
}
.model-status {
  padding: 30px;
  margin-top: 100px;
  font-size: 25px;
  color: var(--color-blue);
  position:relative;
  top:0px;
  opacity: 100;
  z-index:5;
  display: center;
  margin: 0 auto;
}

.trPic{
    position:absolute;
    z-index:2;
    width:100%;
    height:100%;
    background-color: whitesmoke;
    opacity:1;
    text-align: center;
    vertical-align: middle;
    text-opacity: 1;
    float: left;
    top: 40%;
    left: 50%;
    transform: translate(-50%, -50%);
    justify-content:center;
    
	}

	.trPic .redO{

	}
</style>
