from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    SOURCE_STORAGE_CONNECTION_STRING: str = Field(default="")
    DESTINATION_STORAGE_CONNECTION_STRING: str = Field(default="")
    OTEL_SERVICE_NAME: str = Field(default="sara-rain-drop-detector")
    OTEL_EXPORTER_OTLP_ENDPOINT: str = Field(default="http://localhost:4317")
    OTEL_EXPORTER_OTLP_PROTOCOL: str = Field(default="grpc")
    MODEL_PATH: str = Field(default="/app/data/rain_detection_model.pt")
    DETECTION_CONFIDENCE_THRESHOLD: float = Field(default=0.25)


settings = Settings()
