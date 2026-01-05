import uuid
from typing import Any
import hashlib

import uvicorn
from fastapi import FastAPI, status, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import sys

from src import Configuration
from src.use_cases.get_dataset_status import GetDatasetStatusUseCase
from src.use_cases.get_result_pipeline_download import GetResultPipelineDownloadUseCase
from src.use_cases.pipelines_start import PipelineStartDTO, PipelinesStartUseCase

sys.stdout.flush()
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize HTTP Basic authentication mechanism
# This security scheme requires clients to provide a username and password for API access
security = HTTPBasic()

# Create a FastAPI application instance with metadata and security configuration
# The app serves as the core of the API, handling requests and routing with authentication
app = FastAPI(
    title="Democratizing Data",
    version="1.0.0",
    description="API for extracting and processing metadata from OpenAlex API, focusing on dataset usage and related information.",
    dependencies=[Depends(security)]
)

# Define a dependency function to verify HTTP Basic Authentication credentials
# This function checks the provided credentials against the application configuration
def verification(creds: HTTPBasicCredentials = Depends(security)):
    """
    Verify HTTP Basic Authentication credentials against the application configuration.

    This function validates the username and password provided in the HTTP request
    by comparing them with the values stored in the configuration settings.

    Args:
        creds (HTTPBasicCredentials): The credentials (username and password) provided in the request.

    Returns:
        bool: True if the credentials match the configured values.

    Raises:
        HTTPException: 401 Unauthorized error if the credentials are invalid, with a WWW-Authenticate header.
    """
    username = creds.username
    password = creds.password
    config = Configuration().get()
    if (username == config['basic_username'] and password == config['basic_password']):
        return True
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="UNAUTHORIZED",
            headers={"WWW-Authenticate": "Basic"},
        )

@app.post(
    "/pipelines/start",
    tags=["Pipelines"],
    description="Start pipeline to search for publications that use the specified datasets.",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {"description": "Datasets successfully queued for processing"},
        400: {"description": "Bad request or error occurred during pipeline start"},
        401: {"description": "Unauthorized access due to invalid credentials"}
    }
)
async def pipelines_start(
        dataset: PipelineStartDTO,
        Verification=Depends(verification)) -> Any:
    if Verification:
        try:
            hash = hashlib.sha256()
            hash.update(dataset.group.name.encode())
            dataset.group.id = hash.hexdigest()
            pipeline_start = PipelinesStartUseCase()
            task_id = str(uuid.uuid4())
            pipeline_start.execute(task_id, dataset)
            return {'task_id': task_id}
        except Exception as e:
            logger.error(e)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Error on processing datasets, try again"
            )
    return None

@app.get(
    "/pipelines/{task_id}/status",
    tags=["Pipelines"],
    description="Get the status of a pipeline task.",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Datasets found"},
        400: {"description": "Bad request or error occurred"},
        401: {"description": "Unauthorized access due to invalid credentials"}
    }
)
async def pipelines_status(
        task_id: str,
        Verification=Depends(verification)) -> Any:
    if Verification:
        try:
            uuid.UUID(task_id)
            pipeline_status = GetDatasetStatusUseCase()
            result = pipeline_status.execute(task_id)
            return result
        except ValueError as e:
            logger.error(e)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="TaskId invalid, try again"
            )
        except Exception as e:
            logger.error(e)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Error on processing datasets, try again"
            )
    return None

@app.get(
    "/pipelines/{task_id}/result/download",
    tags=["Pipelines"],
    description="Get result of pipeline task.",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Datasets found"},
        400: {"description": "Bad request or error occurred"},
        401: {"description": "Unauthorized access due to invalid credentials"}
    }
)
async def pipelines_result_download(
        task_id: str,
        Verification=Depends(verification)) -> Any:
    if Verification:
        try:
            uuid.UUID(task_id)
            pipeline_result = GetResultPipelineDownloadUseCase()
            result = pipeline_result.execute(task_id)
            return result
        except ValueError as e:
            logger.error(e)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="TaskId invalid, try again"
            )
        except Exception as e:
            logger.error(e)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Error on processing datasets, try again"
            )
    return None

if __name__ == "__main__":
    """
    Entry point for running the FastAPI application directly.

    This block initializes and starts the uvicorn ASGI server with the configured application instance,
    binding it to all network interfaces (0.0.0.0) and using the port specified in the configuration.
    """
    # Retrieve application configuration settings for server parameters
    config = Configuration().get()
    # Start the ASGI server with the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=int(config['port']))