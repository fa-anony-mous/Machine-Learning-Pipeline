import asyncio
import multiprocessing
import uvicorn
import sys
import os

def run_streamlit():
    """Start the Streamlit server in a separate process"""
    import streamlit.web.bootstrap as bootstrap
    from streamlit.web.cli import main as st_main
    sys.argv = ["streamlit", "run", "streamlit_app/app.py"]
    st_main()

async def run_fastapi():
    """Start the FastAPI server"""
    # Add project paths to Python path
    sys.path.insert(0, os.path.abspath('.'))
    
    from streamlit_app.backend.app.main import app as fastapi_app
    config = uvicorn.Config(fastapi_app, host="127.0.0.1", port=8000)
    server = uvicorn.Server(config)
    await server.serve()

async def main():
    """Run both servers"""
    # Start Streamlit in a separate process
    streamlit_process = multiprocessing.Process(target=run_streamlit)
    streamlit_process.start()
    
    # Run FastAPI in the main process
    await run_fastapi()

if __name__ == "__main__":
    asyncio.run(main())