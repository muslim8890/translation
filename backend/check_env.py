
import sys
import traceback

with open("env_check.txt", "w") as f:
    f.write(f"Python: {sys.executable}\n")
    try:
        import uvicorn
        f.write(f"Uvicorn: {uvicorn.__version__}\n")
    except:
        f.write("Uvicorn: MISSING\n")
        f.write(traceback.format_exc() + "\n")

    try:
        import requests
        f.write(f"Requests: {requests.__version__}\n")
    except:
        f.write("Requests: MISSING\n")
        f.write(traceback.format_exc() + "\n")
    
    try:
        import fastapi
        f.write(f"FastAPI: {fastapi.__version__}\n")
    except:
        f.write("FastAPI: MISSING\n")

print("Env Check Done")
