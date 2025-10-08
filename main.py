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
        "image/jpeg", "image/jpg", "image/png", "image/gif",
        "application/pdf", "application/msword", 
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/csv", "text/plain"
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
    description="A Python FastAPI server for file uploads, optimized for Railway deployment"
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
origins = settings.allowed_origins.split(",") if settings.allowed_origins != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]
)

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
        timestamp=datetime.utcnow().isoformat()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Railway monitoring"""
    try:
        import time
        start_time = os.getenv('START_TIME')
        if start_time:
            uptime = time.time() - float(start_time)
        else:
            uptime = 0.0
        
        logger.info(f"Health check requested - uptime: {uptime}")
        
        return HealthResponse(
            status="healthy",
            uptime=uptime,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            uptime=0.0,
            timestamp=datetime.utcnow().isoformat()
        )

@app.get("/api/health", response_model=HealthResponse)
async def api_health_check():
    """API health check endpoint for external monitoring services"""
    try:
        import time
        start_time = os.getenv('START_TIME')
        if start_time:
            uptime = time.time() - float(start_time)
        else:
            uptime = 0.0
        
        logger.info(f"API health check requested - uptime: {uptime}")
        
        return HealthResponse(
            status="healthy",
            uptime=uptime,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"API health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            uptime=0.0,
            timestamp=datetime.utcnow().isoformat()
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
                "timestamp": datetime.utcnow().isoformat()
            }
        
        test_supabase = create_client(supabase_url, supabase_key)
        
        # Test query to get loads for the specific ORG_ID
        org_id = "01970f4c-c79d-7858-8034-60a265d687e4"
        
        result = test_supabase.table("loads").select("id,custom_load_id,status").eq("org_id", org_id).limit(5).execute()
        
        return {
            "message": "Test endpoint working",
            "org_id": org_id,
            "supabase_connection": "success",
            "sample_loads": result.data,
            "total_loads_found": len(result.data),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Test endpoint error: {str(e)}")
        return {
            "message": "Test endpoint error",
            "org_id": "01970f4c-c79d-7858-8034-60a265d687e4",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.post("/upload-multiple", response_model=UploadResponse)
@limiter.limit("5/minute")
async def upload_multiple_files(
    request,
    files: List[UploadFile] = File(...)
):
    """Upload multiple files (max 5)"""
    try:
        if len(files) > 5:
            raise HTTPException(
                status_code=400, 
                detail="Too many files. Maximum 5 files allowed."
            )
        
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        files_info = []
        
        for file in files:
            # Validate file
            if not validate_file_type(file):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid file type for {file.filename}. Only images, documents, and text files are allowed."
                )
            
            if not validate_file_size(file):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} too large. Maximum size is {settings.max_file_size} bytes."
                )
            
            # Read file content
            content = await file.read()
            
            # Create file info
            file_info = FileInfo(
                original_name=file.filename,
                mimetype=file.content_type,
                size=len(content),
                uploaded_at=datetime.utcnow().isoformat()
            )
            
            files_info.append(file_info)
            logger.info(f"File uploaded: {file.filename} ({len(content)} bytes)")
        
        return UploadResponse(
            message=f"{len(files)} files uploaded successfully",
            files=files_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multiple upload error: {str(e)}")
        raise HTTPException(status_code=500, detail="File upload failed")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "Something went wrong" if settings.environment == "production" else str(exc)
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    import time
    os.environ['START_TIME'] = str(time.time())
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
        reload=settings.environment == "development"
    )




def parse_datetime(datetime_str):
    """Parse datetime string and convert to ISO format"""
    if not datetime_str or datetime_str.strip() == "":
        return None

    try:
        # Parse format: "07/13/2025 14:00:00"
        dt = datetime.strptime(datetime_str.strip(), "%m/%d/%Y %H:%M:%S")
        # Do not convert to UTC; store as received
        return dt.isoformat()
    except ValueError:
        return None


def map_equipment_type(equipment_type, source_format="pepsi"):
    """Map equipment types to standard format"""
    if source_format == "pepsi":
        equipment_mapping = {
            "DRY VAN": "Dryvan",
            "REEFER": "Reefer",
            "FLATBED": "Flatbed",
            "POWER ONLY": "Power only",
            "STEP DECK": "Step deck",
        }
    else:  # happyrobot format
        equipment_mapping = {
            "Van": "Van",
            "Dry Van": "Dryvan",
            "Reefer": "Reefer",
            "Flatbed": "Flatbed",
            "Power Only": "Power only",
            "Step Deck": "Step deck",
        }
    return equipment_mapping.get(equipment_type, "Van")


def map_location_type(location_type):
    """Map Pepsi location types to standard format"""
    if "Pick" in location_type:
        return "origin"
    elif "Drop" in location_type:
        return "destination"
    else:
        return "origin"  # default


def normalize_field_names(data):
    """Convert all field names in a dictionary to lowercase"""
    if isinstance(data, dict):
        return {key.lower(): value for key, value in data.items()}
    elif isinstance(data, list):
        return [normalize_field_names(item) for item in data]
    else:
        return data

def map_pepsi_to_happyrobot(happyrobot_load):
    """Map HappyRobot load data to standard load format - handles one load at a time"""
    
    try:
        # Normalize field names to lowercase
        happyrobot_load = normalize_field_names(happyrobot_load)
        print("type of happyrobot_load", type(happyrobot_load))
        print("happyrobot_load", happyrobot_load)
        
        # Parse appointment times
        origin_appointment = happyrobot_load.get("origin_appointment_local", "")
        destination_appointment = happyrobot_load.get("destination_appointment_local", "")
        
        # Convert to ISO format (remove microseconds if present)
        pickup_open = origin_appointment.replace(".000000000", "") if origin_appointment else None
        pickup_close = pickup_open  # Add 2 hours if needed
        delivery_open = destination_appointment.replace(".000000000", "") if destination_appointment else None
        delivery_close = delivery_open  # Add 2 hours if needed
        
        # Create stops
        stops = [
            {
                "type": "origin",
                "location": {
                    "city": happyrobot_load.get("origin_city", ""),
                    "state": happyrobot_load.get("origin_state_code", ""),
                    "zip": str(happyrobot_load.get("origin_postal_code", "")),
                    "country": "US"
                },
                "stop_timestamp_open": pickup_open,
                "stop_timestamp_close": pickup_close,
                "stop_order": 1
            },
            {
                "type": "destination",
                "location": {
                    "city": happyrobot_load.get("destination_city", ""),
                    "state": happyrobot_load.get("destination_state_code", ""),
                    "zip": str(happyrobot_load.get("destination_postal_code", "")),
                    "country": "US"
                },
                "stop_timestamp_open": delivery_open,
                "stop_timestamp_close": delivery_close,
                "stop_order": 2
            }
        ]
        
        # Transform to match your table schema (lowercase field names)
        standard_load = {
            "custom_load_id": happyrobot_load.get("custom_load_id"),
            "equipment_type_name": happyrobot_load.get("equipment_type_name", "Van"),
            "status": "available" if happyrobot_load.get("status", "").upper() == "OPEN" else "unavailable",
            "posted_carrier_rate": happyrobot_load.get("posted_carrier_rate"),
            "max_buy": happyrobot_load.get("max_buy"),
            "type": happyrobot_load.get("type", "OWNED"),
            "weight": happyrobot_load.get("weight"),
            "number_of_pieces": happyrobot_load.get("number_of_pieces"),
            "miles": happyrobot_load.get("miles"),
            "linehaul_rate": None,  # Not provided in HappyRobot data
            "rate_per_mile": None,  # Not provided in HappyRobot data
            "same_day_pickup": None,  # Not provided in HappyRobot data
            "origin_appointment_local": pickup_open,
            "origin_appointment_utc": pickup_open,  # Assuming same as local for now
            "origin_address_1": None,  # Not provided in HappyRobot data
            "origin_city": happyrobot_load.get("origin_city", ""),
            "origin_state_code": happyrobot_load.get("origin_state_code", ""),
            "origin_postal_code": str(happyrobot_load.get("origin_postal_code", "")),
            "destination_appointment_local": delivery_open,
            "destination_appointment_utc": delivery_open,  # Assuming same as local for now
            "destination_address_1": None,  # Not provided in HappyRobot data
            "destination_city": happyrobot_load.get("destination_city", ""),
            "destination_state_code": happyrobot_load.get("destination_state_code", ""),
            "destination_postal_code": str(happyrobot_load.get("destination_postal_code", ""))
        }
        
        return standard_load

    except Exception as e:
        print(f"Error mapping HappyRobot load to standard load: {e}")
        return None



def process_csv_file(csv_file_path):
    """Process the CSV file and convert Pepsi loads to happyrobot format"""
    happyrobot_loads = []

    # Group rows by ORDER_NBR
    loads_by_order = defaultdict(list)

    with open(csv_file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row_num, row in enumerate(reader, 1):
            order_nbr = row.get("ORDER_NBR", "").strip()
            if not order_nbr:
                continue

            loads_by_order[order_nbr].append(row)

    # Process each load
    for order_nbr, rows in loads_by_order.items():
        try:
            # Map to happyrobot format
            print("rows")
            happyrobot_loads_for_order = map_pepsi_to_happyrobot(rows)
            happyrobot_loads.extend(happyrobot_loads_for_order)
            print(f"Processed load {order_nbr} with {len(rows)} stops")

        except Exception as e:
            print(f"Error processing load {order_nbr}: {e}")
            continue

    return happyrobot_loads


@app.post("/upload-to-supabase")
def upload_to_supabase(happyrobot_loads: List[dict]):
    """Upload loads to Supabase after transforming from HappyRobot format"""
    if not happyrobot_loads:
        # if there are no loads to upload, everything is unavailable
        mark_all_loads_unavailable()
        print("No loads to upload")
        return {"message": "No loads provided-- everything unavailable"}

    # Normalize all field names to lowercase
    happyrobot_loads = normalize_field_names(happyrobot_loads)
    
    # Transform loads from HappyRobot format to expected format
    transformed_loads = []
    for load in happyrobot_loads:
        try:
            transformed_load = map_pepsi_to_happyrobot(load)
            transformed_loads.append(transformed_load)
        except Exception as e:
            print(f"Error transforming load {load.get('custom_load_id', 'unknown')}: {e}")
            continue

    if not transformed_loads:
        print("No loads successfully transformed")
        return {"message": "No loads successfully transformed"}

    # Filter out duplicates based on custom_load_id
    seen_load_ids = set()
    unique_loads = []
    print("transformed_loads", transformed_loads)

    

    for load in transformed_loads:
        print("load", load)
        load_id = load.get("custom_load_id")
        print("load_id", load_id)
        if load_id and load_id not in seen_load_ids:
            seen_load_ids.add(load_id)
            unique_loads.append(load)
        elif load_id:
            print(f"Skipping duplicate load_id: {load_id}")

    if not unique_loads:
        print("No unique loads to upload after filtering duplicates")
        return {"message": "No unique loads to upload after filtering duplicates"}

    print(
        f"Uploading {len(unique_loads)} unique loads one at a time (filtered out {len(transformed_loads) - len(unique_loads)} duplicates)"
    )

    # unique loads is unique from upload
    print("unique_loads", unique_loads)
    unique_load_ids = [load["custom_load_id"] for load in unique_loads]
    unique_loads_set = set(unique_load_ids)

    print("unique_loads_set", unique_loads_set)

    # Start from the beginning since this is a new file
    start_index = 0
    total_uploaded = 0
    failed_loads = []

    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
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
                "statusCode": 500,
                "body": json.dumps(f"Supabase credentials not found in environment (ENVIRONMENT={environment})")
            }
        
        supabase = create_client(supabase_url, supabase_key)

        # Get today's date in YYYY-MM-DD format
        today = datetime.now().strftime("%Y-%m-%d")
        # Add time component for proper comparison with ISO format
        today_start = f"{today}T00:00:00Z"

        existing_loads = (
            supabase.table("loads")
            .select("custom_load_id")
            # .gte("origin_appointment_local", today_start)
            .eq("status", "available")
            .execute()
        )


        # Create a set of existing load IDs for quick lookup
        existing_load_ids = {load["custom_load_id"] for load in existing_loads.data}
        print("existing_load_ids", existing_load_ids)
        print(existing_loads.data)


        loads_to_update = existing_load_ids - unique_loads_set
        print("loads_to_update", loads_to_update)
        print(f"Loads present in DB but not in CSV: {loads_to_update}")
        if loads_to_update and loads_to_update != set():
            # Update status to "unavailable" for loads not in current response
            update_result = (
                supabase.table("loads")
                .update({"status": "unavailable"})
                .in_("custom_load_id", list(loads_to_update))
                .execute()
            )
            print(f"Updated {len(loads_to_update)} loads to unavailable status")

        # upsert loads one by one since we don't have the upsert_loads function
        successful_uploads = 0
        failed_uploads = 0

        for load in unique_loads:
            try:
                print(f"Uploading load {load['custom_load_id']}")
                
                # Try to insert first
                result = supabase.table("loads").insert(load).execute()
                successful_uploads += 1
                print(f"Successfully inserted load {load['custom_load_id']}")

            except Exception as e:
                # If insert fails, try to update
                try:
                    print(f"Insert failed for {load['custom_load_id']}, trying update: {str(e)}")
                    result = supabase.table("loads").update(load).eq("custom_load_id", load['custom_load_id']).execute()
                    successful_uploads += 1
                    print(f"Successfully updated load {load['custom_load_id']}")
                except Exception as update_error:
                    failed_uploads += 1
                    print(f"Error uploading load {load['custom_load_id']}: {str(update_error)}")
                    continue

        print(
            f"Upload complete: {successful_uploads} successful, {failed_uploads} failed"
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                f"Successfully processed {successful_uploads} loads, {failed_uploads} failed. "
                f"Marked {len(loads_to_update)} loads as unavailable"
            ),
        }

    except Exception as e:
        print(f"Error uploading to Supabase: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps(
                f"Error uploading to Supabase: {e}"
            ),
        }


# def main():
#     csv_file_path = (
#         "test_loads.csv"  ## This is the only line that has to be edited Kabir
#     )

#     if not os.path.exists(csv_file_path):
#         print(f"CSV file not found: {csv_file_path}")
#         return

#     print("Processing PepsiCo loads...")
#     happyrobot_loads = process_csv_file(csv_file_path)

#     print(f"\nProcessed {len(happyrobot_loads)} loads")

#     # Save to JSON file for review
#     with open("pepsi_loads_converted.json", "w") as f:
#         json.dump(happyrobot_loads, f, indent=2)
#     print("Saved converted loads to pepsi_loads_converted.json")

#     # Upload to Supabase
#     print("\nUploading to Supabase...")
#     upload_to_supabase(happyrobot_loads)


def mark_all_loads_unavailable():
    try:
        print("Marking all loads as unavailable")
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
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
                "statusCode": 500,
                "body": json.dumps(f"Supabase credentials not found in environment (ENVIRONMENT={environment})")
            }
        
        supabase = create_client(supabase_url, supabase_key)


        existing_loads = (
            supabase.table("loads")
            .select("custom_load_id")
            .eq("status", "available")
            .execute()
        )
        print("existing_loads data=", existing_loads.data)
        
        if existing_loads.data and len(existing_loads.data) > 0:
            # Extract just the load IDs from the response
            existing_load_ids = [load["custom_load_id"] for load in existing_loads.data]
            print("existing_load_ids", existing_load_ids)
            
            # Update status to "unavailable" for all available loads
            update_result = (
                supabase.table("loads")
                .update({"status": "unavailable"})
                .in_("custom_load_id", existing_load_ids)
                .execute()
            )
            print(f"Updated {len(existing_load_ids)} loads to unavailable status")
    except Exception as e:
        print(f"Error marking all loads as unavailable: {e}")
        return