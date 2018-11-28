<template>
  <v-dialog v-model="show" max-width="600px">
    <v-card>
      <v-card-title><div class="info-panel-title">More Information</div></v-card-title>
      <v-card-text>
        <div class="info-panel-text" v-if="currentView === 'resnet50'">
          <p>Note that ~100 MB of weights must be loaded.</p><p>Enter any valid image URL as input to the network. You can also select from a list of prepopulated image URLs. The endpoint must have CORS enabled, to enable us to extract the numeric data from the canvas element, so not all URLs will work. Imgur and <a target="_blank" rel="noopener noreferrer" href="https://www.flickr.com/search/?text=&license=2%2C3%2C4%2C5%2C6%2C9&sort=interestingness-desc">Flickr creative commons</a> all work, and are good places to start. After running the network, the top-5 classes are displayed.</p><p>All computation performed entirely in your browser. Toggling GPU on should offer significant speedups compared to CPU.</p>
        </div>
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn color="primary" flat @click.stop="show = false">Close</v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script lang="ts">
import {Watch, Vue, Component, Prop} from 'vue-property-decorator';

@Component({
  // props: {
  //   showInfoPanel: { type: Boolean, default: false },
  //   currentView: { type: String },
  //   close: { type: Function },
  // },
})

export default class InfoPanel extends Vue {
  isShow: boolean;
  constructor() {
    super();
    this.isShow = false;
  }

  @Prop({default: false}) isShowInfoPanel!: boolean;
  @Prop(String) currentView!: string;
  @Prop(Function) close!: void;

  @Watch('') 
  showInfoPanel(newVal: boolean) {
    this.isShow = newVal;
  }

  @Watch('')
  show(newVal: boolean) {
    // if (!newVal) {
    //   this.close();
    // }
  }
}
</script>

<style scoped lang="postcss">
@import '../variables.css';

.info-panel-title {
  margin-top: 10px;
  font-size: 14px;
  font-weight: 600;
  color: var(--color-lightgray);
}

.info-panel-text {
  font-size: 14px;
  color: var(--color-darkgray);

  & a {
    color: var(--color-blue);
    transition: color 0.2s ease-in;

    &:hover {
      color: var(--color-blue-light);
    }
  }
}
</style>
