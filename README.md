# 실행 방법

### .env

- MONGO_URL=YOUR_URL
- DB_NAME=YOUR_DBNAME
- COLLECTION_NAME=YOUR_COLLECTIONNAME

### backend

- cmd창을 열고
  > cd backend -> pip install -r requirements.txt -> uvicorn main:app --reload 순으로 입력

### frontend

- 새로운 cmd창을 열고
  > cd frontend -> npm install -> node server.js 순으로 입력

> http://localhost:3000로 들어가서 이미지 파일 업로드로 시작

# API 설명

### post "/inference"

- 업로드한 파일이 이미지가 맞는 지 유효성 검사
- 업로드된 이미지를 backend/uploads 폴더에 uuid로 파일이름을 바꿔서 저장
- 모델 추론 후 top3 결과를 DB에 저장
- id, model_name, topk를 화면에 출력하기 위해 결과 반환

### get "/inference"

- 몇번째부터 볼건지 skip을 통해 지정, 최대 제한 10으로 지정, 지금까지 분석한 이미지 총 개수 출력
- find로 skip, limit에 따른 목록 불러오기(id, model_name, original_filename, topk[0])

### get "/inference/{id}"

- 존재하는 id인지 검증
- id는 있으나 결과 없을 시 에러 반환
- id를 통해 하나의 데이터 가져오기
- 화면에 출력(id, model_name, original_filename, created_at, topk)
