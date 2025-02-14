import tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';

const 온도 = [20, 21, 22, 23];
const 판매량 = [40, 42, 44, 46];

const 원인 = tf.tensor(온도).reshape([온도.length, 1]);
const 결과 = tf.tensor(판매량).reshape([판매량.length, 1]);

const X = tf.input({ shape: [1] });
const Y = tf.layers.dense({ units: 1 }).apply(X);
const model = tf.model({ inputs: X, outputs: Y as tf.SymbolicTensor });

model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
})

model.fit(원인, 결과, { epochs: 3000 }).then((result) => {
    const 예측한결과 = model.predict(원인) as tf.Tensor;
    예측한결과.print();
    model.save('file://./model-1a'); 
});

console.log("Hello via Bun!");
