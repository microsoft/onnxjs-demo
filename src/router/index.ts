import Vue from 'vue';
import Router from 'vue-router';
import Home from '../components/Home.vue';
import Resnet50 from '../components/models/Resnet50.vue';
import SqueezeNet from '../components/models/Squeezenet.vue';
// import Yolo from '../components/models/Yolo.vue';
import Emotion from '../components/models/Emotion.vue';
import MNIST from '../components/models/MNIST.vue';

Vue.use(Router);

export default new Router({
  mode: 'history',
  base: process.env.BASE_URL,
  routes: [
    {
      path: '*',
      name: 'home',
      component: Home,
    },
    {
      path: '/resnet50',
      component: Resnet50,
    },
    {
      path: '/squeezenet',
      component: SqueezeNet,
    },
    // {
    //   path: '/yolo',
    //   component: Yolo,
    // },
    {
      path:'/emotion',
      component: Emotion,
    },
    {
      path:'/mnist',
      component: MNIST,
    }
    // { 
    //   path: '/', 
    //   redirect: '*'
    // }
  ],
});
