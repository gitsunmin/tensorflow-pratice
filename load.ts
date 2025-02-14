import tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';

console.time('load');

await tf.loadLayersModel('file://./model-1a/model.json').then((model) => {
    const result = model.predict(tf.tensor([20, 50, 100, 150, 90000]));
    if (Array.isArray(result)) {
        result.forEach(t => t.print());
    } else {
        result.print();
    }
    console.timeEnd('load');
})