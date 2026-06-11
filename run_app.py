import subprocess
import time
import sys
import os

def main():
    print("Starting LangGraph backend server...")
    
    # Start the LangGraph backend process
    try:
        backend_process = subprocess.Popen(["langgraph", "dev"])
    except FileNotFoundError:
        # Fallback for Windows in case it's a .cmd script without an .exe wrapper
        try:
            backend_process = subprocess.Popen(["langgraph.cmd", "dev"])
        except FileNotFoundError:
            print("Error: 'langgraph' command not found. Ensure it is installed and in your PATH.")
            sys.exit(1)

    print("Waiting 5 seconds for backend to initialize on port 2024...")
    time.sleep(5)

    print("Starting Streamlit frontend...")
    # Start the Streamlit frontend process
    try:
        frontend_process = subprocess.Popen(
            ["streamlit", "run", "src/app.py", "--server.address", "0.0.0.0", "--server.port", "8502"]
        )
    except FileNotFoundError:
        try:
            frontend_process = subprocess.Popen(
                ["streamlit.cmd", "run", "src/app.py", "--server.address", "0.0.0.0", "--server.port", "8502"]
            )
        except FileNotFoundError:
            print("Error: 'streamlit' command not found. Ensure it is installed and in your PATH.")
            backend_process.terminate()
            sys.exit(1)

    try:
        print("\nBoth servers are running!")
        print("Frontend: http://localhost:8502")
        print("Backend:  http://localhost:2024")
        print("Press Ctrl+C to stop both servers.\n")
        
        # Keep the main thread alive and monitor the subprocesses
        while True:
            time.sleep(1)
            # Exit if either process terminates unexpectedly
            if backend_process.poll() is not None:
                print("LangGraph backend exited unexpectedly.")
                break
            if frontend_process.poll() is not None:
                print("Streamlit frontend exited unexpectedly.")
                break
                
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down gracefully...")
        
    finally:
        # Ensure both processes are terminated
        if frontend_process and frontend_process.poll() is None:
            print("Stopping Streamlit frontend...")
            frontend_process.terminate()
            frontend_process.wait()
            
        if backend_process and backend_process.poll() is None:
            print("Stopping LangGraph backend...")
            backend_process.terminate()
            backend_process.wait()
            
        print("All processes stopped successfully.")

if __name__ == "__main__":
    main()
