import subprocess

packages = [
    "streamlit", "tensorflow", "scikit-image", "seaborn", "opencv-python-headless",
    "google-generativeai", "huggingface-hub", "faiss-cpu", "unstructured",
    "langchain-google-genai", "sentence-transformers", "python-dotenv",
    "pypdf2", "Pillow", "langchain-community"
]

for pkg in packages:
    try:
        version = subprocess.check_output(["pip", "show", pkg], text=True)
        for line in version.splitlines():
            if line.startswith("Version:"):
                print(f"{pkg}=={line.split(' ')[1]}")
    except subprocess.CalledProcessError:
        print(f"{pkg} is NOT installed.")
