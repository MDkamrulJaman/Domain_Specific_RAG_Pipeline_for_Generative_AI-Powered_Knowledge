from pydantic import BaseModel


class IngestResponse(BaseModel):
    message: str
    chunks: int