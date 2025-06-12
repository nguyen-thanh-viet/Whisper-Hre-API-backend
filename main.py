from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client
import tempfile
from gradio_client import Client, file as gr_file

app = FastAPI()

# Cấu hình CORS để frontend có thể gọi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # hoặc thay * bằng domain frontend cụ thể
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Client("ntviet/whisper-small-hre5.2")

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/transcribe/")
async def transcribe(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # result = client.predict(tmp_path)
    # Sử dụng gradio_client.file để tạo đúng metadata
    result = client.predict(gr_file(tmp_path))
    return {"transcription": result}
