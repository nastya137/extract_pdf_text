from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    replicate_api_token: str = ""
    enable_ocr: bool = True

    external_service_url: str = "http://localhost:8001/process"

    max_pdf_size_mb: int = 50
    max_pages: int = 500

    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"