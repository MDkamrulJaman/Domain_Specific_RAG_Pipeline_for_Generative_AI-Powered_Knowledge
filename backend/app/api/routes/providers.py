from fastapi import APIRouter
from app.schemas.chat import Provider
from app.services.provider_service import provider_configuration, inspect_provider

providers_router = APIRouter(prefix="/providers", tags=["Providers"])


@providers_router.get("/{provider}")
def provider_status(provider: Provider, check_connection: bool = False):
    return inspect_provider(provider) if check_connection else provider_configuration(provider)
