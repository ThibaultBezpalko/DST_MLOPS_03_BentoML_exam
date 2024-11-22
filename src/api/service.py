import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, JSON
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
from datetime import datetime, timedelta
import os


# Secret key and algorithm for JWT authentication
JWT_SECRET_KEY = "This_is_not_a_secret_key"
JWT_ALGORITHM = "HS256"

# User credentials for authentication
USERS = {
    "user123": "password123",
    "user456": "password456"
}

class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path == "/v1/models/admissions/predict":
            token = request.headers.get("Authorization")
            if not token:
                return JSONResponse(status_code=401, content={"detail": "Missing authentication token"})

            try:
                token = token.split()[1]  # Remove 'Bearer ' prefix
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return JSONResponse(status_code=401, content={"detail": "Token has expired"})
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})

            request.state.user = payload.get("sub")

        response = await call_next(request)
        return response

# Pydantic model to validate input data
class InputModel(BaseModel):
    serial_nb: int
    gre: int
    toefl: int
    university_rating: int
    sop: float
    lor: float
    cgpa: float
    research: int

# Get the model from the Model Store
admissions_runner = bentoml.sklearn.get("admissions:latest").to_runner()

# Create a service API
admissions_service = bentoml.Service("admissions_service", runners=[admissions_runner])

# Add the JWTAuthMiddleware to the service
admissions_service.add_asgi_middleware(JWTAuthMiddleware)

# Create an API endpoint for the service
@admissions_service.api(input=JSON(), output=JSON())
def login(credentials: dict) -> dict:
    username = credentials.get("username")
    password = credentials.get("password")

    if username in USERS and USERS[username] == password:
        token = create_jwt_token(username)
        return {"token": token}
    else:
        return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})

# Create an API endpoint for the service
@admissions_service.api(
    input=JSON(pydantic_model=InputModel),
    output=JSON(),
    route='v1/models/admissions/predict'
)
async def classify(input_data: InputModel, ctx: bentoml.Context) -> dict:
    request = ctx.request
    user = request.state.user if hasattr(request.state, 'user') else None

    # Convert the input data to a numpy array
    input_series = np.array([
        input_data.serial_nb, input_data.gre, input_data.toefl, input_data.university_rating,
        input_data.sop, input_data.lor, input_data.cgpa, input_data.research
    ])

    result = await admissions_runner.predict.async_run(input_series.reshape(1, -1))

    return {
        "prediction": result.tolist(),
        "user": user
    }

# Function to create a JWT token
def create_jwt_token(user_id: str):
    expiration = datetime.utcnow() + timedelta(hours=1)
    payload = {
        "sub": user_id,
        "exp": expiration
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token