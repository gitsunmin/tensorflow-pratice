import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';
import { goodsNameList } from './goodsNameList';

const tokenize = (data: string[]) => {
    const charTokenizer = new Map<string, number>();
    const reverseCharTokenizer = new Map();
    let charIndex = 1;

    // 🔹 글자 단위로 매핑
    for (const name of goodsNameList) {
        for (const char of name) {  // ✨ 단어가 아니라 글자 단위로 반복
            if (!charTokenizer.has(char)) {
                charTokenizer.set(char, charIndex);
                reverseCharTokenizer.set(charIndex, char);
                charIndex++;
            }
        }
    }
    return {
        charTokenizer,
        reverseCharTokenizer
    }
};

const textToCharSequence = (text: string, charTokenizer: Map<string, number>) => {
    return text.split("").map(char => charTokenizer.get(char) || 0);
}

const createTrainingData = (data: string[], charTokenizer: Map<string, number>) => {
    let inputSequences: number[][] = [];
    let outputChars: number[] = [];

    for (const name of data) {
        let tokenList = textToCharSequence(name, charTokenizer);
        for (let i = 1; i < tokenList.length; i++) {
            inputSequences.push(tokenList.slice(0, i));
            outputChars.push(tokenList[i]);
        }
    }

    return { inputSequences, outputChars };
};

const applyPadding = (inputSequences: number[][], maxLen: number) => {
    return inputSequences.map(seq => {
        const padding = Array(maxLen - seq.length).fill(0);
        return [...padding, ...seq];
    });
};

const convertToTensor = (inputSequences: number[][], outputChars: number[], maxLen: number) => {
    const XArray = applyPadding(inputSequences, maxLen);
    const X = tf.tensor2d(XArray, [XArray.length, maxLen], 'float32');
    const y = tf.tensor1d(outputChars, 'float32');

    return { X, y };
};

const createModel = (vocabSize: number, maxLen: number) => {
    const model = tf.sequential();

    model.add(tf.layers.embedding({
        inputDim: vocabSize,
        outputDim: 16,  // 기존 10 → 16으로 증가
        inputLength: maxLen
    }));

    model.add(tf.layers.lstm({ units: 128, returnSequences: true })); // 기존 50 → 128 유닛 증가
    model.add(tf.layers.lstm({ units: 64, returnSequences: false })); // 추가 LSTM 레이어

    model.add(tf.layers.dense({ units: vocabSize, activation: 'softmax' }));

    return model;
};

const trainModel = async (model: tf.Sequential, X: tf.Tensor, y: tf.Tensor) => {
    model.compile({
        optimizer: tf.train.adam(0.005),  // 기존보다 2배 증가 (기본값: 0.001)
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy']
    });

    console.log("📌 모델 학습 시작...");
    await model.fit(X, y, {
        epochs: 200,
        batchSize: 8,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                if (logs) {
                    console.log(`Epoch ${epoch + 1}: loss=${logs.loss}, accuracy=${logs.acc}`);
                }
            }
        }
    });

    console.log("📌 모델 학습 완료!");
};

const predictNextChar = (model: tf.Sequential, seedText: string, charTokenizer: Map<string, number>, reverseCharTokenizer: Map<number, string>, maxLen: number) => {
    let inputSeq = textToCharSequence(seedText, charTokenizer);
    const padding = Array(maxLen - inputSeq.length).fill(0);
    const paddedInput = [...padding, ...inputSeq];

    const inputTensor = tf.tensor2d([paddedInput], [1, maxLen], 'float32');
    const prediction = model.predict(inputTensor) as tf.Tensor;
    // 🔹 예측된 확률값 확인
    console.log("📌 예측된 확률 분포:", prediction.arraySync());
    // 🔹 가장 높은 확률을 가진 인덱스 찾기
    const predictedIndex = prediction.argMax(1).dataSync()[0];

    return reverseCharTokenizer.get(predictedIndex) || "예측 실패";
};

// 🔹 토큰화 수행
const { charTokenizer, reverseCharTokenizer } = tokenize(goodsNameList);
const vocabSize = charTokenizer.size + 1;

// 🔹 학습 데이터 생성
const { inputSequences, outputChars } = createTrainingData(goodsNameList, charTokenizer);
const maxLen = inputSequences.reduce((max, seq) => Math.max(max, seq.length), 0);
// 🔹 패딩 및 텐서 변환
const { X, y } = convertToTensor(inputSequences, outputChars, maxLen);

// 🔹 모델 생성 및 학습
const model = createModel(vocabSize, maxLen);
await trainModel(model, X, y);

// 🔹 예측 테스트
console.log("예측 결과:", predictNextChar(model, "유기", charTokenizer, reverseCharTokenizer, maxLen));
console.log("예측 결과:", predictNextChar(model, "국내", charTokenizer, reverseCharTokenizer, maxLen));
console.log("예측 결과:", predictNextChar(model, "오메", charTokenizer, reverseCharTokenizer, maxLen));