import subprocess
import time
import requests
import sys
import os

def run_test():
    print("🚀 Starting Microservice Smoke Test...")
    
    # Start the service in the background
    # We use sys.executable to ensure we use the same environment
    proc = subprocess.Popen([sys.executable, "service/main.py"], 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
    
    time.sleep(10) # Wait for startup
    
    try:
        # 1. Health Check
        print("🔍 Checking /health...")
        h = requests.get("http://localhost:8000/health")
        print(f"Health Response: {h.json()}")
        
        # 2. Prediction Test
        print("🔍 Checking /predict...")
        p = requests.post("http://localhost:8000/predict", 
                          json={"url": "http://google.com"})
        print(f"Predict Response: {p.json()}")
        
        if h.status_code == 200 and p.status_code == 200:
            print("✅ Smoke Test Passed!")
        else:
            print("❌ Smoke Test Failed!")
            
    except Exception as e:
        print(f"❌ Error during smoke test: {e}")
    finally:
        print("🛑 Terminating service...")
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=5)
            # print(stdout.decode())
        except:
            proc.kill()

if __name__ == "__main__":
    run_test()
