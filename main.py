from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from app.utils import save_upload_file
from app.ai import detect_text, predict_banknote_class , assistant_response , detect_objects_with_yolo
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel



app = FastAPI()

@app.post("/detect-text/")
async def detect_text_from_image(file: UploadFile = File(...)):
    # Save image locally
    image_path = await save_upload_file(file)
    print(f" Received file saved at: {image_path}") 
    # Run OCR
    detected_text = detect_text(image_path)
    print(f" Detected Text: {detected_text}")
    return JSONResponse(content={"text": detected_text})



@app.post("/detect-banknote/")
async def detect_banknote(file: UploadFile = File(...)):
    image_path = await save_upload_file(file)
    print(f" Received banknote image at: {image_path}")
    result = predict_banknote_class(image_path)
    return JSONResponse(content=result)

#hh

# CORS (to connect with Flutter)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask-assistant/")
def ask_ai(req: QuestionRequest):
    answer = assistant_response(req.question)
    return {"answer": answer}




@app.post("/detect-objects/")
async def detect_objects(file: UploadFile = File(...)):
    image_path = await save_upload_file(file)
    print(f" Image for object detection saved at: {image_path}")
    result = detect_objects_with_yolo(image_path)
    return JSONResponse(content=result)
