import os
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
import httpx
from app.extractor import extract_text
from app.settings import Settings

settings = Settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF Text Extractor Service",
    version="1.0.0",
    swagger_js_url="/static/swagger-ui-bundle.js",
    swagger_css_url="/static/swagger-ui.css",
)

# Монтируем папку со статикой — в Dockerfile мы скачали файлы в /app/static
app.mount("/static", StaticFiles(directory="/app/static"), name="static")


@app.post("/extract")
async def extract_and_forward(
    file: UploadFile = File(..., description="PDF file to extract text from")
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are allowed")
    content = await file.read()
    if not content.startswith(b'%PDF'):
        raise HTTPException(400, "Invalid PDF file header")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        logger.info(f"Processing {file.filename} (temp file: {tmp_path})")
        extracted_text = extract_text(tmp_path)
        logger.info(f"Extracted {len(extracted_text)} characters")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                settings.external_service_url,
                json={"text": extracted_text, "source": file.filename}
            )
            response.raise_for_status()
            external_result = response.json()

        return {
            "status": "success",
            "filename": file.filename,
            "text_length": len(extracted_text),
            "external_service_response": external_result
        }

    except Exception as e:
        logger.exception("Error during processing")
        raise HTTPException(500, f"Processing failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.debug(f"Removed temp file {tmp_path}")


@app.get("/health")
async def health_check():
    return {"status": "ok"}