# Bridge Load Upload Server

A Python FastAPI server designed for Railway deployment with file upload capabilities.

## Features

- 🚀 **Railway Ready**: Optimized for Railway deployment
- 📁 **File Upload**: Single and multiple file upload support
- 🔒 **Security**: CORS, rate limiting, and input validation
- 📊 **Logging**: Comprehensive logging with Python logging
- 🏥 **Health Check**: Built-in health check endpoint
- ⚡ **Performance**: FastAPI with async/await support
- 📚 **Auto Documentation**: Automatic OpenAPI/Swagger documentation

## API Endpoints

### Base Routes
- `GET /` - Server information and status
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

### Upload Routes
- `POST /upload` - Upload a single file
- `POST /upload-multiple` - Upload multiple files (max 5)

## Local Development

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   cp env.local .env
   ```

3. **Start development server:**
   ```bash
   python main.py
   ```
   
   Or with uvicorn directly:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 3000
   ```

4. **Access the application:**
   - Server: `http://localhost:3000`
   - API Docs: `http://localhost:3000/docs`
   - ReDoc: `http://localhost:3000/redoc`

## Railway Deployment

### Method 1: Railway CLI

1. **Install Railway CLI:**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login to Railway:**
   ```bash
   railway login
   ```

3. **Initialize Railway project:**
   ```bash
   railway init
   ```

4. **Deploy:**
   ```bash
   railway up
   ```

### Method 2: GitHub Integration

1. **Push your code to GitHub**

2. **Connect Railway to your GitHub repository:**
   - Go to [Railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

3. **Configure environment variables:**
   - In Railway dashboard, go to Variables tab
   - Add the following variables:
     ```
     ENVIRONMENT=production
     ALLOWED_ORIGINS=https://yourdomain.com
     ```

4. **Deploy:**
   - Railway will automatically deploy on every push to main branch

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Environment mode | `development` |
| `PORT` | Server port | `3000` |
| `ALLOWED_ORIGINS` | Comma-separated list of allowed origins | `*` |

## File Upload

### Single File Upload
```bash
curl -X POST \
  -F "file=@/path/to/your/file.jpg" \
  http://localhost:3000/upload
```

### Multiple File Upload
```bash
curl -X POST \
  -F "files=@/path/to/file1.jpg" \
  -F "files=@/path/to/file2.pdf" \
  http://localhost:3000/upload-multiple
```

### Python Example
```python
import requests

# Single file upload
with open('test.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:3000/upload', files=files)
    print(response.json())

# Multiple file upload
files = [
    ('files', open('file1.jpg', 'rb')),
    ('files', open('file2.pdf', 'rb'))
]
response = requests.post('http://localhost:3000/upload-multiple', files=files)
print(response.json())
```

## Supported File Types

- Images: JPEG, JPG, PNG, GIF
- Documents: PDF, DOC, DOCX, XLS, XLSX, CSV, TXT

## Security Features

- **CORS**: Cross-origin resource sharing
- **Rate Limiting**: 10 requests per minute for single upload, 5 for multiple
- **File Validation**: Type and size validation
- **Trusted Host**: Host validation middleware
- **Input Sanitization**: Pydantic model validation

## Project Structure

```
BridgeLoadUpload/
├── main.py              # Main FastAPI server file
├── requirements.txt     # Python dependencies
├── railway.toml         # Railway configuration
├── env.example          # Environment variables template
├── env.local            # Local development environment
├── .gitignore           # Git ignore file
└── README.md            # This file
```

## API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: Available at `/docs` endpoint
- **ReDoc**: Available at `/redoc` endpoint
- **OpenAPI Schema**: Available at `/openapi.json` endpoint

## Dependencies

- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server implementation
- **Pydantic**: Data validation using Python type annotations
- **SlowAPI**: Rate limiting for FastAPI
- **Python-multipart**: File upload support
- **Aiofiles**: Async file operations

## Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   # Kill process using port 3000
   lsof -ti:3000 | xargs kill -9
   ```

2. **File upload fails:**
   - Check file size (max 10MB)
   - Verify file type is supported
   - Ensure proper multipart/form-data encoding

3. **Railway deployment fails:**
   - Check Railway logs in dashboard
   - Verify all environment variables are set
   - Ensure `railway.toml` is properly configured
   - Check Python version compatibility

4. **Import errors:**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

## Performance Tips

- FastAPI uses async/await for better performance
- File uploads are handled asynchronously
- Rate limiting prevents abuse
- Automatic request validation with Pydantic

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

## License

MIT License - see LICENSE file for details
