import tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';


await tf.loadLayersModel('file://./model-1a/model.json').then((model) => {
    model.predict(tf.tensor([20])).print();
})