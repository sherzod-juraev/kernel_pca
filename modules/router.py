from fastapi import APIRouter, status
from .scheme import KernelPCAIn, KernelPCAOut


modules_router = APIRouter()


@modules_router.post(
    '/',
    summary='Kernel principal component analysis',
    status_code=status.HTTP_200_OK,
    response_model=KernelPCAOut
)
async def kpca_fit(
        X: KernelPCAIn
) -> KernelPCAOut:
    pass