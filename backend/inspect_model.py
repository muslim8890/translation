try:
    path = "d:/app_file/translat/backend/models/nllb-200-ct2/model.bin"
    import os
    
    with open("inspect_result.txt", "w") as log:
        if os.path.exists(path):
            size = os.path.getsize(path)
            log.write(f"Size: {size}\n")
            with open(path, "rb") as f:
                content = f.read(500)
                log.write(f"Content: {content}\n")
        else:
             log.write("File not found\n")

except Exception as e:
    with open("inspect_result.txt", "w") as log:
        log.write(f"Error: {e}\n")

