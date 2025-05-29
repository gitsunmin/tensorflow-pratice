import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';
import { ProductNameClassifier } from './traniner';

/**
 * 저장된 모델을 불러와 상품명 예측 테스트
 */
export async function testProductNameClassifier(): Promise<void> {
    try {
        // 모델 인스턴스 생성
        const classifier = new ProductNameClassifier();

        // 저장된 모델 불러오기
        const modelDir = path.join(__dirname, 'src', 'modules', 'word-predicte', 'models', 'product-name-classifier');
        const modelJsonPath = path.join(modelDir, 'model.json');

        if (!fs.existsSync(modelJsonPath)) {
            console.log('모델을 찾을 수 없습니다. 먼저 모델을 학습시켜야 합니다:');
            console.log('  node -r ts-node/register src/modules/word-predicte/traniner.ts');
            console.log('  bun src/modules/word-predicte/traniner.ts');
            return;
        }

        console.log('이전에 학습된 모델을 불러오는 중...');
        await classifier.loadModel();

        // 테스트할 텍스트 예시
        const testCases = [
            "★초특가★ 흙 대파 1단 (국내산)",
            "백오이 10kg (상,약 50개,국내산)BOX",
            "안녕하세요 반갑습니다",
            "주문해주셔서 감사합니다",
            "양파 슬라이스 냉동",
            "마늘 다진 국내산 1kg",
            "오늘 날씨가 좋네요",
            "상품에 대해 문의드립니다",
            // 추가 테스트 케이스
            "신선한 사과 5kg 박스",
            "환불 요청합니다",
        ];

        // 예측 실행 및 결과 출력
        console.log('\nPrediction results:');
        for (const text of testCases) {
            const result = classifier.predict(text);
            console.log(`"${text}" => ${result.isProduct ? '상품명' : '비상품명'} (confidence: ${result.confidence.toFixed(4)})`);
        }

    } catch (error) {
        console.error('모델 테스트 중 오류 발생:', error);
    }
}

// 모듈이 직접 실행될 때 테스트 시작
if (require.main === module) {
    testProductNameClassifier();
}