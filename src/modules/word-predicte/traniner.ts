import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';

// 상품명 데이터 가져오기 (실제 구현 시 경로 조정 필요)
import { koreanProductNames } from '@/data/korean-product-name-list';

/**
 * 단어를 문자 시퀀스로 변환하는 토크나이저 클래스
 */
class CharTokenizer {
    private tokenToIndex: Map<string, number> = new Map();
    private indexToToken: Map<number, string> = new Map();
    private vocabSize: number = 0;
    public maxLen: number;

    constructor(maxLen: number = 100) {
        this.maxLen = maxLen;
    }

    /**
     * 텍스트 데이터셋으로 토크나이저 학습
     */
    fit(texts: string[]): void {
        // 특수 토큰 추가
        this.tokenToIndex.set('<PAD>', 0);
        this.indexToToken.set(0, '<PAD>');

        // 모든 고유 문자 찾기
        let index = 1;
        for (const text of texts) {
            for (const char of text) {
                if (!this.tokenToIndex.has(char)) {
                    this.tokenToIndex.set(char, index);
                    this.indexToToken.set(index, char);
                    index++;
                }
            }
        }

        this.vocabSize = this.tokenToIndex.size;
        console.log(`Vocabulary size: ${this.vocabSize}`);
    }

    /**
     * 텍스트를 인덱스 시퀀스로 변환
     */
    textToSequence(text: string): number[] {
        const sequence = [];
        for (const char of text) {
            sequence.push(this.tokenToIndex.get(char) || 0);
        }
        return sequence;
    }

    /**
     * 텍스트 시퀀스 패딩 처리
     */
    padSequences(sequences: number[][]): number[][] {
        return sequences.map(seq => {
            if (seq.length >= this.maxLen) {
                return seq.slice(0, this.maxLen);
            }
            return [...seq, ...Array(this.maxLen - seq.length).fill(0)];
        });
    }

    /**
     * 텍스트 목록을 패딩된 시퀀스로 변환
     */
    textsToSequences(texts: string[]): number[][] {
        const sequences = texts.map(text => this.textToSequence(text));
        return this.padSequences(sequences);
    }

    /**
     * 현재 어휘 크기 반환
     */
    getVocabSize(): number {
        return this.vocabSize;
    }

    /**
     * 토크나이저 저장
     */
    save(filepath: string): void {
        try {
            // indexToToken이 Map인지 확인
            if (!(this.indexToToken instanceof Map)) {
                this.indexToToken = new Map(this.indexToToken);
            }

            const data = {
                indexToToken: Array.from(this.indexToToken.entries()),
                vocabSize: this.vocabSize,
                maxLen: this.maxLen
            };

            // 디렉토리 존재 여부 확인
            const path = require('path');
            const dir = path.dirname(filepath);
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
            }

            fs.writeFileSync(filepath, JSON.stringify(data));
            console.log('모델이 성공적으로 저장되었습니다.');
        } catch (error) {
            console.error('모델 저장 중 오류 발생:', error);
            throw error;
        }
    }

    /**
     * 토크나이저 로드
     */
    load(filepath: string): void {
        const content = fs.readFileSync(filepath, 'utf-8');
        const data = JSON.parse(content);

        this.tokenToIndex = new Map(data.tokenToIndex);
        this.indexToToken = new Map(data.indexToToken);
        this.vocabSize = data.vocabSize;
        this.maxLen = data.maxLen;

        console.log(`Tokenizer loaded from ${filepath}`);
    }
}

/**
 * 데이터 준비 및 전처리
 */
class DataProcessor {
    private positiveExamples: string[];
    private negativeExamples: string[] = [];

    constructor(
        private productNames: string[],
        private tokenizer: CharTokenizer,
        private testSplit: number = 0.2
    ) {
        this.positiveExamples = this.productNames.filter(name => name.trim().length > 0);
        this.generateNegativeExamples();
    }

    /**
     * 부정 예제(비상품명) 생성
     */
    private generateNegativeExamples(): void {
        // 1. 상품명을 섞어서 새로운 텍스트 생성
        for (let i = 0; i < this.positiveExamples.length / 2; i++) {
            const idx1 = Math.floor(Math.random() * this.positiveExamples.length);
            const idx2 = Math.floor(Math.random() * this.positiveExamples.length);

            if (idx1 !== idx2) {
                const text1 = this.positiveExamples[idx1];
                const text2 = this.positiveExamples[idx2];

                // 두 상품명의 일부를 조합
                const part1 = text1.substring(0, Math.floor(text1.length / 2));
                const part2 = text2.substring(Math.floor(text2.length / 2));

                this.negativeExamples.push(part1 + part2);
            }
        }

        // 2. 상품명 글자 순서 변경
        for (let i = 0; i < this.positiveExamples.length / 2; i++) {
            const idx = Math.floor(Math.random() * this.positiveExamples.length);
            const text = this.positiveExamples[idx];

            if (text.length > 3) {
                // 글자 순서 변경
                const chars = text.split('');
                for (let j = 0; j < chars.length / 2; j++) {
                    const randIdx1 = Math.floor(Math.random() * chars.length);
                    const randIdx2 = Math.floor(Math.random() * chars.length);

                    if (randIdx1 !== randIdx2) {
                        const temp = chars[randIdx1];
                        chars[randIdx1] = chars[randIdx2];
                        chars[randIdx2] = temp;
                    }
                }

                this.negativeExamples.push(chars.join(''));
            }
        }

        // 3. 일반 문장 또는 무작위 문자열 추가 (실제 구현 시 확장 가능)
        const commonPhrases = [
            "안녕하세요", "반갑습니다", "좋은 하루 되세요",
            "주문해주셔서 감사합니다", "배송이 지연됩니다",
            "결제가 완료되었습니다", "문의 감사합니다",
            "이벤트 참여 방법", "구매 후기 작성하기"
        ];

        this.negativeExamples.push(...commonPhrases);

        console.log(`Generated ${this.negativeExamples.length} negative examples`);
    }

    /**
     * 훈련 및 테스트 데이터 준비
     */
    prepareData(): {
        trainX: tf.Tensor2D;
        trainY: tf.Tensor1D;
        testX: tf.Tensor2D;
        testY: tf.Tensor1D;
    } {
        // 모든 텍스트 데이터 결합
        const allTexts = [...this.positiveExamples, ...this.negativeExamples];
        const allLabels = [
            ...Array(this.positiveExamples.length).fill(1),
            ...Array(this.negativeExamples.length).fill(0)
        ];

        // 데이터 셔플
        const indices = Array.from(Array(allTexts.length).keys());
        tf.util.shuffle(indices);

        const shuffledTexts = indices.map(i => allTexts[i]);
        const shuffledLabels = indices.map(i => allLabels[i]);

        // 토크나이저 학습
        this.tokenizer.fit(shuffledTexts);

        // 텍스트를 시퀀스로 변환
        const sequences = this.tokenizer.textsToSequences(shuffledTexts);

        // 훈련/테스트 분할
        const splitIdx = Math.floor(sequences.length * (1 - this.testSplit));

        const trainSequences = sequences.slice(0, splitIdx);
        const testSequences = sequences.slice(splitIdx);

        const trainLabels = shuffledLabels.slice(0, splitIdx);
        const testLabels = shuffledLabels.slice(splitIdx);

        // 텐서 변환
        const trainX = tf.tensor2d(trainSequences, [trainSequences.length, this.tokenizer.maxLen], 'float32');
        const trainY = tf.tensor1d(trainLabels, 'float32');

        const testX = tf.tensor2d(testSequences, [testSequences.length, this.tokenizer.maxLen], 'float32');
        const testY = tf.tensor1d(testLabels, 'float32');

        return { trainX, trainY, testX, testY };
    }
}

/**
 * 상품명 분류 모델
 */
export class ProductNameClassifier {
    private model!: tf.Sequential;
    private tokenizer: CharTokenizer;
    private modelSavePath: string;

    constructor(
        private maxLen: number = 100,
        private embeddingDim: number = 64,
        modelName: string = 'product-name-classifier'
    ) {
        this.tokenizer = new CharTokenizer(maxLen);

        // 절대 경로 사용
        const rootDir = path.resolve(__dirname, '../../../');
        const modelDir = path.join(rootDir, 'models', modelName);
        
        console.log('Model directory:', modelDir);

        // 디렉토리 존재 확인 및 생성
        if (!fs.existsSync(modelDir)) {
            fs.mkdirSync(modelDir, { recursive: true });
            console.log(`Created directory: ${modelDir}`);
        }

        // 모델 저장 경로 설정
        this.modelSavePath = `file://${modelDir}`;
    }

    /**
     * 모델 구축
     */
    private buildModel(): tf.Sequential {
        const model = tf.sequential();

        // 임베딩 레이어
        model.add(tf.layers.embedding({
            inputDim: 1000, // 초기값, fit() 호출 시 업데이트
            outputDim: this.embeddingDim,
            inputLength: this.maxLen,
            maskZero: true,
        }));

        // 양방향 LSTM 레이어
        model.add(tf.layers.bidirectional({
            layer: tf.layers.lstm({ units: 64, returnSequences: true }),
            mergeMode: 'concat'
        }));

        // 두 번째 LSTM 레이어
        model.add(tf.layers.bidirectional({
            layer: tf.layers.lstm({ units: 32 }),
            mergeMode: 'concat'
        }));

        // 드롭아웃
        model.add(tf.layers.dropout({ rate: 0.5 }));

        // 완전 연결 레이어
        model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
        model.add(tf.layers.dropout({ rate: 0.3 }));

        // 출력 레이어
        model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

        // 모델 컴파일
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });

        return model;
    }

    /**
     * 모델 학습
     */
    async train(
        productNames: string[],
        epochs: number = 20,
        batchSize: number = 32,
        validationSplit: number = 0.2
    ): Promise<void> {
        // 데이터 처리기 생성
        const dataProcessor = new DataProcessor(productNames, this.tokenizer, validationSplit);
        const { trainX, trainY, testX, testY } = dataProcessor.prepareData();

        // 임베딩 레이어 업데이트
        const vocabSize = this.tokenizer.getVocabSize();
        const newEmbedding = tf.layers.embedding({
            inputDim: vocabSize,
            outputDim: this.embeddingDim,
            inputLength: this.maxLen,
            maskZero: true,
        });

        // 모델 재구축
        this.model = this.buildModel();

        // 모델 재컴파일
        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });

        console.log('Model training started...');

        // 모델 학습
        await this.model.fit(trainX, trainY, {
            epochs,
            batchSize,
            validationData: [testX, testY],
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}/${epochs} - loss: ${logs?.loss.toFixed(4)} - acc: ${logs?.acc.toFixed(4)} - val_loss: ${logs?.val_loss.toFixed(4)} - val_acc: ${logs?.val_acc.toFixed(4)}`);
                }
            }
        });

        console.log('Model training completed');

        // 토크나이저 저장
        const modelDir = this.modelSavePath.replace('file://', '');
        const tokenizerPath = path.join(modelDir, 'tokenizer.json');

        // 토크나이저 저장 전 기존 파일 확인 및 삭제
        if (fs.existsSync(tokenizerPath)) {
            fs.unlinkSync(tokenizerPath);
        }

        this.tokenizer.save(tokenizerPath);

        // 모델 저장 (오류 처리 개선)
        try {
            console.log(`Attempting to save model to ${this.modelSavePath}`);
            const saveResult = await this.model.save(this.modelSavePath);
            console.log('Model save result:', saveResult);
            
            // 파일이 실제로 생성되었는지 확인
            const modelDir = this.modelSavePath.replace('file://', '');
            const modelJsonPath = path.join(modelDir, 'model.json');
            
            if (fs.existsSync(modelJsonPath)) {
                console.log(`Model file successfully created at: ${modelJsonPath}`);
            } else {
                console.error(`Model file was NOT created at: ${modelJsonPath}`);
            }
        } catch (error) {
            console.error('Error saving model:', error);
            throw error;
        }

        // 테스트 데이터로 평가
        const evalResult = this.model.evaluate(testX, testY) as tf.Scalar[];
        console.log(`Test loss: ${evalResult[0].dataSync()[0].toFixed(4)}`);
        console.log(`Test accuracy: ${evalResult[1].dataSync()[0].toFixed(4)}`);

        // 메모리 정리
        trainX.dispose();
        trainY.dispose();
        testX.dispose();
        testY.dispose();
    }

    /**
     * 모델 로드
     */
    async loadModel(): Promise<void> {
        try {
            // 모델 로드 - model.json 파일 명시적 지정
            const modelJsonPath = `${this.modelSavePath}/model.json`;
            console.log(`Loading model from ${modelJsonPath}`);
            this.model = await tf.loadLayersModel(modelJsonPath) as tf.Sequential;
            console.log(`Model loaded from ${modelJsonPath}`);

            // 토크나이저 로드
            const modelDir = this.modelSavePath.replace('file://', '');
            const tokenizerPath = path.join(modelDir, 'tokenizer.json');

            console.log('tokenizerPath:, ', tokenizerPath);
            this.tokenizer.load(tokenizerPath);
        } catch (error) {
            console.error('Error loading model:', error);
            throw error;
        }
    }

    /**
     * 텍스트가 상품명인지 예측
     */
    predict(text: string): { isProduct: boolean; confidence: number } {
        // 텍스트를 시퀀스로 변환
        const sequence = this.tokenizer.textsToSequences([text]);

        // 텐서 변환
        const inputTensor = tf.tensor2d(sequence, [1, this.maxLen], 'float32');

        // 예측
        const prediction = this.model.predict(inputTensor) as tf.Tensor;
        const confidence = prediction.dataSync()[0];

        // 메모리 정리
        inputTensor.dispose();
        prediction.dispose();

        return {
            isProduct: confidence > 0.5,
            confidence
        };
    }
}

/**
 * 메인 함수
 */
export async function trainProductNameClassifier(): Promise<void> {
    try {
        // 상품명 필터링 (빈 문자열 제거)
        const filteredNames = koreanProductNames.filter(name => name.trim().length > 0);

        // 모델 인스턴스 생성
        const classifier = new ProductNameClassifier();

        // 모델 학습
        await classifier.train(filteredNames, 10, 32, 0.2);

        // 모델 테스트
        const testCases = [
            "★초특가★ 흙 대파 1단 (국내산)", // 실제 상품명
            "백오이 10kg (상,약 50개,국내산)BOX", // 실제 상품명
            "안녕하세요 반갑습니다", // 일반 문장
            "주문해주셔서 감사합니다", // 일반 문장
            "양파 슬라이스 냉동", // 상품명 형태
            "마늘 다진 국내산 1kg", // 상품명 형태
            "오늘 날씨가 좋네요", // 일반 문장
            "상품에 대해 문의드립니다" // 일반 문장
        ];

        console.log('\nPrediction test:');
        for (const text of testCases) {
            const result = classifier.predict(text);
            console.log(`"${text}" => ${result.isProduct ? '상품명' : '비상품명'} (confidence: ${result.confidence.toFixed(4)})`);
        }

    } catch (error) {
        console.error('Error training model:', error);
    }
}

// 모듈이 직접 실행될 때 학습 시작
if (require.main === module) {
    trainProductNameClassifier();
}