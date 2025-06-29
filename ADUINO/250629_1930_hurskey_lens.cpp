// 허스키렌즈와 아두이노를 활용한 간단단 프로젝트 예시

// AI 자율주행 자동차: 허스키렌즈로 길을 인식하고 자동차가 스스로 주행
// 어린이 보호구역 시스템: 허스키렌즈로 사람을 인식해서 안전 시스템을 작동
// 네오픽셀 & 부저 제어: 허스키렌즈가 특정 물체를 인식하면 네오픽셀 LED가 켜지거나 부저 울림

// 허스키 렌즈 테스트 코드

if (!huskylens.request()) Serial.println(F("Fail to request data from HUSKYLENS, recheck the connection!"));
else if(!huskylens.isLearned()) Serial.println(F("Nothing learned, press learn button on HUSKYLENS to learn one!"));
else if(!huskylens.available()) Serial.println(F("No block or arrow appears on the screen!"));
else
{
    Serial.println(F("###########"));
    while (huskylens.available())
    {
        HUSKYLENSResult result = huskylens.read();
        printResult(result);
    }    
}


// 화살표 가져오기

// 블록이나 화살표 가져오기
HUSKYLENSResult get(int16_t index)

// 특정 ID의 블록이나 화살표 가져오기
HUSKYLENSResult get(int16_t ID, int16_t index)

// 블록 가져오기
HUSKYLENSResult getBlock(int16_t index)

// 특정 ID의 블록 가져오기
HUSKYLENSResult getBlock(int16_t ID, int16_t index)



// Threshold(임계값)

// Confidence Threshold(신뢰도 임계값): 모델이 탐지한 객체가 실제로 그 객체일 확률이 얼마나 되는지를 나타내는 값  
// ex) confidence threshold = 0.5 -> 모델이 50% 이상의 확률로 "이건 ㅇㅇ"라고 판단한 경우에만 그 객체를 ㅇㅇ로 인식
// IoU(Intersection over Union) Threshold: 예측한 경계 상자(bounding box)와 실제 객체의 위치가 얼마나 겹치는지를 나타내는 값
// ex) IoU가 1이면 완벽하게 겹침, 0이면 전혀 겹치지 않음. 보통 0.5 이상이면 같은 객체를 탐지했다고 판단

// Threshold가 너무 높으면: 확실한 객체만 탐지하게 되어 정확도(precision)는 높아지지만, 많은 객체를 놓칠 수 있음(recall이 낮아짐)
// Threshold가 너무 낮으면: 더 많은 객체를 탐지할 수 있지만(recall이 높아짐), 잘못된 탐지(false positive)도 많아져서 정확도가 떨어질 수 있음

// 허스키 렌즈 confidence threshold 설정
huskylens.setConfidenceThreshold(60); // 60% 이상의 확률로 탐지된 객체만 인식

// 자율주행 자동차 프로젝트를 가정 
// 1) 사람을 탐지하는 경우: 안전이 중요하니까 confidence threshold를 낮게 설정해서 사람을 놓치지 않도록 함
// 2)특정 표지판을 탐지하는 경우: 정확한 판단이 필요하니까 confidence threshold를 높게 설정

// 결론
// Threshold 설정은 결국 precision(정확도)과 recall(재현율) 사이의 트레이드오프 
// 이 둘 사이의 최적점을 찾기 위해 다양한 threshold 값으로 테스트

