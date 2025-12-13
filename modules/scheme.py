from pydantic import BaseModel, field_validator
from numpy import array, nan, nanmean, take, where, isnan
from fastapi import HTTPException, status


class KernelPCAOut(BaseModel):

    X: list[list]


class KernelPCAIn(BaseModel):
    model_config = {
        'extra': 'forbid'
    }

    X: list[list]

    @field_validator('X')
    def verify_X(cls, value):
        X = array([[nan if col is None else col for col in row] for row in value])
        if X.ndim != 2:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='X must be 2D matrix'
            )
        col_mean = nanmean(X, axis=0)
        idx = where(isnan(X))
        X[idx] = take(col_mean, idx[1])
        return X