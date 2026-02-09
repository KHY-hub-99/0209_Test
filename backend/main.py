import os, json, ast
import uuid
import shutil
import requests
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from bson import ObjectId
from pydantic import BaseModel, ConfigDict

import torch
from torch.nn import functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB 설정
MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

try:
    client = MongoClient(MONGO_URL)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    client.admin.command('ping')
    print(f"MongoDB 연결 완료!")
except Exception as e:
    print(f"MongoDB 연결 중 오류 : {e}")

# 이미지 저장 폴더 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
print(f"이미지 저장 경로: {UPLOAD_DIR}")

# 라벨데이터
imagenet_labels = {}
LABELS_PATH = os.path.join(BASE_DIR, "imagenet1000_clsidx_to_labels.txt")

try:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
        imagenet_labels = ast.literal_eval(content)
    print(f"라벨 딕셔너리 로드 완료 (총 {len(imagenet_labels)}개)")
except FileNotFoundError:
    print(f"경고: {LABELS_PATH} 파일을 찾을 수 없습니다.")
except SyntaxError:
    print("경고: 파일 형식이 올바른 Python 딕셔너리가 아닙니다.")

# 모델 로드(eval만)
try:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    model.eval()
    transform = weights.transforms()
    print("모델로드 완료")
except Exception as e:
    print(f"모델로드 중 오류 : {e}")

# Pydantic 모델
class TopKItem(BaseModel):
    label: str
    score: float

class InferenceResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    id: str
    model_name: str
    topk: List[TopKItem]

class InferenceListResponse(BaseModel):
    items: List[dict]
    skip: int
    limit: int
    count: int

# 헬퍼 함수
def serialize_doc(doc):
    if doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return doc

# API 설정
@app.post("/inference", response_model=InferenceResponse)
async def predict(file: UploadFile=File(...), cnn_model_name: str=Form("efficientnet_b0", alias="model_name"), top_k: int=Form(3)):

    # 파일 검증
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드하세요!")

    # 파일 저장
    file_ext = file.filename.split(".")[-1]
    saved_filename = f"{uuid.uuid4()}.{file_ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_filename)

    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 저장 중 오류 : {e}")
    
    # 모델 추론
    try:
        image = Image.open(saved_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs[0], dim=0)
        
        # top 3개 추출 및 리스트에 저장
        top_prob, top_catid = torch.topk(probabilities, k=top_k)

        topk_results = []
        for i in range(top_k):
            idx = top_catid[i].item()
            score = round(top_prob[i].item(), 4)
            label = imagenet_labels.get(idx, imagenet_labels.get(str(idx), "Unknown"))
            topk_results.append({"label": label, "score": score})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 중 오류 : {str(e)}")
    
    # DB 저장
    doc = {
        "original_filename": file.filename,
        "saved_path": saved_path,
        "model_name": cnn_model_name,
        "topk": topk_results,
        "created_at": datetime.now()
    }

    result = collection.insert_one(doc)

    # 결과 반환
    return {
        "id": str(result.inserted_id),
        "model_name": cnn_model_name,
        "topk": topk_results
    }

@app.get("/inference/{id}")
def get_inference(id: str):
    if not ObjectId.is_valid(id):
        raise HTTPException(status_code=400, detail="유효하지 않은 ID 형식입니다!")
    
    doc = collection.find_one({"_id": ObjectId(id)})
    if not doc:
        raise HTTPException(status_code=404, detail="결과를 찾을 수 없습니다.")
    return serialize_doc(doc)

@app.get("/inference", response_model=InferenceListResponse)
def get_inference_list(skip: int=0, limit: int=10, cnn_model_name: Optional[str]=None):

    query = {}
    if cnn_model_name:
        query["model_name"] = cnn_model_name

    total_count = collection.count_documents(query)
    cursor = collection.find(query).skip(skip).limit(limit).sort("created_at", -1)
    items = [serialize_doc(doc) for doc in cursor]

    return {
        "items": items,
        "skip": skip,
        "limit": limit,
        "count": total_count
    }