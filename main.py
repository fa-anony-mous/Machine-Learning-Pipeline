import sys
import os

def run_streamlit():
    """Start the Streamlit server"""
    import streamlit.web.bootstrap as bootstrap
    from streamlit.web.cli import main as st_main
    
    # Add project paths to Python path
    sys.path.insert(0, os.path.abspath('.'))
    
    # Run the Streamlit app
    sys.argv = ["streamlit", "run", "streamlit_app/app.py"]
    st_main()

if __name__ == "__main__":
    run_streamlit()