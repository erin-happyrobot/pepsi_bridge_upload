from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import os
import aiofiles
from datetime import datetime
from typing import List, Optional
import logging
import json
from collections import defaultdict
from datetime import datetime, timedelta
from supabase import create_client, Client
import time
import csv
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Supabase client (will be created in test endpoint)
supabase: Client = None


# Settings
class Settings(BaseSettings):
    app_name: str = "Bridge Load Upload Server"
    version: str = "1.0.0"
    environment: str = "development"
    port: int = 3000
    allowed_origins: str = "*"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: List[str] = [
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/gif",
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/csv",
        "text/plain",
    ]
    # Supabase settings
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    base_url: Optional[str] = None
    api_key_bridge_load_upload: Optional[str] = None
    org_id: Optional[str] = None

    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra environment variables


settings = Settings()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="A Python FastAPI server for file uploads, optimized for Railway deployment",
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
origins = (
    settings.allowed_origins.split(",") if settings.allowed_origins != "*" else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])


# Pydantic models
class ServerInfo(BaseModel):
    message: str
    version: str
    status: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    uptime: float
    timestamp: str


class FileInfo(BaseModel):
    original_name: str
    mimetype: str
    size: int
    uploaded_at: str


class UploadResponse(BaseModel):
    message: str
    file: Optional[FileInfo] = None
    files: Optional[List[FileInfo]] = None


# Utility functions
def validate_file_type(file: UploadFile) -> bool:
    """Validate if file type is allowed"""
    return file.content_type in settings.allowed_file_types


def validate_file_size(file: UploadFile) -> bool:
    """Validate if file size is within limits"""
    return file.size <= settings.max_file_size


# Routes
@app.get("/", response_model=ServerInfo)
async def root():
    """Root endpoint with server information"""
    return ServerInfo(
        message=settings.app_name,
        version=settings.version,
        status="running",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Railway monitoring"""
    try:
        import time

        start_time = os.getenv("START_TIME")
        if start_time:
            uptime = time.time() - float(start_time)
        else:
            uptime = 0.0

        logger.info(f"Health check requested - uptime: {uptime}")

        return HealthResponse(
            status="healthy", uptime=uptime, timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy", uptime=0.0, timestamp=datetime.utcnow().isoformat()
        )


@app.get("/api/health", response_model=HealthResponse)
async def api_health_check():
    """API health check endpoint for external monitoring services"""
    try:
        import time

        start_time = os.getenv("START_TIME")
        if start_time:
            uptime = time.time() - float(start_time)
        else:
            uptime = 0.0

        logger.info(f"API health check requested - uptime: {uptime}")

        return HealthResponse(
            status="healthy", uptime=uptime, timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"API health check error: {e}")
        return HealthResponse(
            status="unhealthy", uptime=0.0, timestamp=datetime.utcnow().isoformat()
        )


@app.get("/test")
async def test_endpoint():
    """Test endpoint with ORG_ID"""
    try:
        # Load environment variables
        from dotenv import load_dotenv

        load_dotenv()

        # Test Supabase connection
        # Check environment and use appropriate credentials
        environment = os.getenv("ENVIRONMENT", "production")
        if environment.upper() == "DEV":
            supabase_url = os.getenv("SUPABASE_URL_TEST")
            supabase_key = os.getenv("SUPABASE_KEY_TEST")
            print("Using TEST Supabase credentials for DEV environment")
        else:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY")
            print("Using PRODUCTION Supabase credentials")

        if not supabase_url or not supabase_key:
            return {
                "message": "Environment variables not set",
                "org_id": "01970f4c-c79d-7858-8034-60a265d687e4",
                "error": f"Supabase credentials not found in environment (ENVIRONMENT={environment})",
                "timestamp": datetime.utcnow().isoformat(),
            }

        test_supabase = create_client(supabase_url, supabase_key)

        # Test query to get loads for the specific ORG_ID
        org_id = "01970f4c-c79d-7858-8034-60a265d687e4"

        result = (
            test_supabase.table("loads")
            .select("id,custom_load_id,status")
            .eq("org_id", org_id)
            .limit(5)
            .execute()
        )

        return {
            "message": "Test endpoint working",
            "org_id": org_id,
            "supabase_connection": "success",
            "sample_loads": result.data,
            "total_loads_found": len(result.data),
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Test endpoint error: {str(e)}")
        return {
            "message": "Test endpoint error",
            "org_id": "01970f4c-c79d-7858-8034-60a265d687e4",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": (
                "Something went wrong"
                if settings.environment == "production"
                else str(exc)
            ),
        },
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    import time

    os.environ["START_TIME"] = str(time.time())
    logger.info(f"Starting {settings.app_name} v{settings.version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Port: {settings.port}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down server")


if __name__ == "__main__":
    import uvicorn

    # Use PORT environment variable for Railway deployment
    port = int(os.getenv("PORT", settings.port))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.environment == "development",
    )


def normalize_field_names(data):
    """Convert all field names in a dictionary to lowercase"""
    if isinstance(data, dict):
        return {key.lower(): value for key, value in data.items()}
    elif isinstance(data, list):
        return [normalize_field_names(item) for item in data]
    else:
        return data


def transform_happyrobot_load(load):
    """Transform unformatted HappyRobot load to standard format"""
    try:
        # Normalize field names to lowercase
        load = normalize_field_names(load)

        # Parse appointment times - remove microseconds
        origin_appointment_local = load.get("origin_appointment_local", "")
        origin_appointment_utc = load.get("origin_appointment_utc", "")
        destination_appointment_local = load.get("destination_appointment_local", "")
        destination_appointment_utc = load.get("destination_appointment_utc", "")

        pickup_open = (
            origin_appointment_local.replace(".000000000", "")
            if origin_appointment_local
            else None
        )
        pickup_open_utc = (
            origin_appointment_utc.replace(".000000000", "")
            if origin_appointment_utc
            else None
        )
        delivery_open = (
            destination_appointment_local.replace(".000000000", "")
            if destination_appointment_local
            else None
        )
        delivery_open_utc = (
            destination_appointment_utc.replace(".000000000", "")
            if destination_appointment_utc
            else None
        )

        # Create origin and destination objects
        origin = {
            "address_1": load.get("origin_address_1", ""),
            "city": load.get("origin_city", ""),
            "state": load.get("origin_state_code", ""),
            "zip": str(load.get("origin_postal_code", "")),
            "country": "US",
        }

        destination = {
            "address_1": load.get("destination_address_1", ""),
            "city": load.get("destination_city", ""),
            "state": load.get("destination_state_code", ""),
            "zip": str(load.get("destination_postal_code", "")),
            "country": "US",
        }

        # Create stops array
        stops = [
            {
                "type": "origin",
                "location": origin,
                "stop_timestamp_open": pickup_open,
                "stop_timestamp_close": pickup_open,  # Same as open for now
                "stop_order": 1,
            },
            {
                "type": "destination",
                "location": destination,
                "stop_timestamp_open": delivery_open,
                "stop_timestamp_close": delivery_open,  # Same as open for now
                "stop_order": 2,
            },
        ]

        # Create standard load object
        standard_load = {
            "org_id": "01970f4c-c79d-7858-8034-60a265d687e4",
            "custom_load_id": load.get("custom_load_id"),
            "equipment_type_name": load.get("equipment_type_name", "Van"),
            "status": (
                "available"
                if load.get("status", "").upper() == "OPEN"
                else "unavailable"
            ),
            "posted_carrier_rate": load.get("posted_carrier_rate"),
            "type": load.get("type", "owned").lower(),
            "is_partial": False,
            "origin": origin,
            "destination": destination,
            "stops": stops,
            "max_buy": load.get("max_buy"),
            "sale_notes": None,
            "commodity_type": load.get("commodity_type", ""),
            "weight": load.get("weight"),
            "number_of_pieces": load.get("number_of_pieces"),
            "miles": load.get("miles"),
            "dimensions": None,
            "pickup_date_open": pickup_open,
            "pickup_date_close": pickup_open,
            "delivery_date_open": delivery_open,
            "delivery_date_close": delivery_open,
            "temp_configuration": None,
            "min_temp": None,
            "max_temp": None,
            "is_temp_metric": False,
            "cargo_value": None,
            "is_hazmat": False,
            "is_owned": True,
        }

        return standard_load

    except Exception as e:
        print(f"Error transforming load: {e}")
        return None


@app.post("/upload-to-supabase")
def upload_to_supabase(happyrobot_loads: List[dict]):
    """Upload loads to Supabase after transforming from HappyRobot format"""
    if not happyrobot_loads:
        print("No loads to upload")
        return {"statusCode": 400, "body": json.dumps("No loads provided")}

    # Transform loads to standard format
    transformed_loads = []
    for load in happyrobot_loads:
        try:
            transformed_load = transform_happyrobot_load(load)
            if transformed_load:
                transformed_loads.append(transformed_load)
        except Exception as e:
            print(
                f"Error transforming load {load.get('custom_load_id', 'unknown')}: {e}"
            )
            continue

    if not transformed_loads:
        print("No loads successfully transformed")
        return {
            "statusCode": 400,
            "body": json.dumps("No loads successfully transformed"),
        }

    # Filter out duplicates based on custom_load_id
    seen_load_ids = set()
    unique_loads = []

    for load in transformed_loads:
        load_id = load.get("custom_load_id")
        if load_id and load_id not in seen_load_ids:
            seen_load_ids.add(load_id)
            unique_loads.append(load)
        elif load_id:
            print(f"Skipping duplicate load_id: {load_id}")

    if not unique_loads:
        print("No unique loads to upload after filtering duplicates")
        return {
            "statusCode": 400,
            "body": json.dumps("No unique loads to upload after filtering duplicates"),
        }

    print(
        f"Uploading {len(unique_loads)} unique loads (filtered out {len(transformed_loads) - len(unique_loads)} duplicates)"
    )

    try:
        # Load environment variables
        from dotenv import load_dotenv

        load_dotenv()

        # Check environment and use appropriate credentials
        environment = os.getenv("ENVIRONMENT", "production")
        if environment.upper() == "DEV":
            supabase_url = os.getenv("SUPABASE_URL_TEST")
            supabase_key = os.getenv("SUPABASE_KEY_TEST")
            print("Using TEST Supabase credentials")
        else:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY")
            print("Using PRODUCTION Supabase credentials")

        if not supabase_url or not supabase_key:
            return {
                "statusCode": 500,
                "body": json.dumps(
                    f"Supabase credentials not found (ENVIRONMENT={environment})"
                ),
            }

        supabase = create_client(supabase_url, supabase_key)

        # Get existing loads from Supabase that are currently available
        org_id = "01970f4c-c79d-7858-8034-60a265d687e4"
        existing_loads = (
            supabase.table("loads")
            .select("custom_load_id")
            .eq("status", "available")
            .eq("org_id", org_id)
            .execute()
        )

        # Create a set of existing load IDs for quick lookup
        existing_load_ids = {load["custom_load_id"] for load in existing_loads.data}
        print(f"Found {len(existing_load_ids)} existing available loads in database")

        # Create a set of current load IDs from incoming request
        current_load_ids = {load.get("custom_load_id") for load in unique_loads}

        # Find loads that exist in Supabase but not in current request
        loads_to_mark_covered = existing_load_ids - current_load_ids

        if loads_to_mark_covered:
            # Update status to "covered" for loads not in current request
            print(
                f"Marking {len(loads_to_mark_covered)} loads as covered (not in current request)"
            )
            supabase.table("loads").update({"status": "covered"}).in_(
                "custom_load_id", list(loads_to_mark_covered)
            ).eq("org_id", org_id).execute()
            print(f"✓ Updated {len(loads_to_mark_covered)} loads to covered status")

        # Upload loads one at a time using rpc.upsert_loads
        total_uploaded = 0
        failed_loads = []

        for i, load in enumerate(unique_loads, 1):
            load_id = load.get("custom_load_id", "unknown")
            print(f"Uploading load {i}/{len(unique_loads)}: {load_id}")

            try:
                supabase.rpc("upsert_loads", {"_payload": [load]}).execute()
                total_uploaded += 1
                print(f"✓ Successfully uploaded load {load_id}")
                time.sleep(0.5)  # Small delay between uploads

            except Exception as e:
                print(f"✗ Failed to upload load {load_id}: {e}")
                failed_loads.append({"load_id": load_id, "error": str(e)})

        print(f"\nUpload Summary:")
        print(f"Successfully uploaded: {total_uploaded} loads")
        print(f"Failed uploads: {len(failed_loads)} loads")
        print(f"Marked as covered: {len(loads_to_mark_covered)} loads")

        if failed_loads:
            print(f"\nFailed loads:")
            for failed in failed_loads:
                print(f"  - {failed['load_id']}: {failed['error']}")

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": f"Successfully uploaded {total_uploaded} loads, {len(failed_loads)} failed, {len(loads_to_mark_covered)} marked as covered",
                    "total_uploaded": total_uploaded,
                    "failed_count": len(failed_loads),
                    "covered_count": len(loads_to_mark_covered),
                    "failed_loads": failed_loads,
                }
            ),
        }

    except Exception as e:
        print(f"Error uploading to Supabase: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error uploading to Supabase: {e}"),
        }
