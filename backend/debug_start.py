import sys
import traceback
import os

print(f"Python executable: {sys.executable}")
print(f"CWD: {os.getcwd()}")

try:
    print("Attempting to import main...")
    import main
    print("Successfully imported main.")
    
    import uvicorn
    print("Starting Uvicorn on port 8000...")
    uvicorn.run(main.app, host="0.0.0.0", port=8000)
    print("Uvicorn exited normally.")
    
except Exception:
    print("CRASHED!")
    traceback.print_exc()
