from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    APP_NAME: str = "PhishingDetection"
    REGISTRY_PATH: str = "model_registry"
    PRODUCTION_STAGE: str = "production"
    TIMEOUT: float = 2.0
settings = Settings()
