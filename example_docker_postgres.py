import json
import modal
import argparse
import time

app = modal.App("dejavu-fingerprinting")
image = modal.Image.debian_slim(python_version="3.9").env(
    {"DEBIAN_FRONTEND": "noninteractive"}
).apt_install(
    "git",
    "gcc",
    "nano",
    "ffmpeg",
    "libasound-dev",
    "portaudio19-dev",
    "libportaudio2",
    "libportaudiocpp0",
    "postgresql",
    "postgresql-contrib",
    "libsndfile1",
    "python3-dev",
    "build-essential",
    "wget",
    "libgomp1",
).run_commands(
    "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
    "dpkg -i cuda-keyring_1.1-1_all.deb",
    "apt-get update",
    "apt-get install -y cuda-toolkit-12-0",
).pip_install(
    "numpy",
    "scipy",
    "matplotlib",
    "pydub",
    "pyaudio",
    "psycopg2-binary",
    "cupy-cuda12x==12.0.0",
    "pybind11",
).env(
    {"CUDA_HOME": "/usr/local/cuda-12.0",
     "PATH": "/usr/local/cuda-12.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
     "LD_LIBRARY_PATH": "/usr/local/cuda-12.0/lib64:/usr/local/lib:/usr/lib:/lib"}
).add_local_dir("dejavu", remote_path="/root/dejavu"
).add_local_dir("mp3", remote_path="/root/mp3"
).add_local_dir("nostalgia", remote_path="/root/nostalgia"
).add_local_dir("test", remote_path="/root/test")

gpu = "A100-40GB"

def setup_postgres():
    import subprocess
    import os
    import glob
    
    os.environ['TZ'] = 'US/Pacific'
    
    subprocess.run(["service", "postgresql", "start"])
    
    time.sleep(5)
    
    pg_hba_path = glob.glob("/etc/postgresql/*/main/pg_hba.conf")[0]
    
    with open(pg_hba_path, "a") as f:
        f.write("local   all             all                                     md5\n")
        f.write("host    all             all             127.0.0.1/32            md5\n")
    
    subprocess.run(["service", "postgresql", "restart"])
    time.sleep(5)
    
    subprocess.run(["psql", "-U", "postgres", "-c", "ALTER USER postgres WITH PASSWORD 'postgres';"])
    
    subprocess.run(["createdb", "-U", "postgres", "-E", "UTF8", "-T", "template0", "dejavu"])
    
    subprocess.run(["psql", "-U", "postgres", "-d", "dejavu", "-c", "ALTER DATABASE dejavu SET timezone TO 'US/Pacific';"])

def process_audio_files_local():
    from dejavu import Dejavu
    from dejavu.logic.recognizer.file_recognizer import FileRecognizer
    
    config = {
        "database": {
            "host": "localhost",
            "user": "postgres",
            "password": "postgres",
            "database": "dejavu"
        },
        "database_type": "postgres"
    }

    djv = Dejavu(config)

    try:
        djv.fingerprint_directory("/root/test", [".wav"])

        results = djv.recognize(FileRecognizer, "/root/mp3/Josh-Woodward--I-Want-To-Destroy-Something-Beautiful.mp3")
        print(f"From file we recognized: {results}\n")

        recognizer = FileRecognizer(djv)
        results = recognizer.recognize_file("/root/mp3/Josh-Woodward--I-Want-To-Destroy-Something-Beautiful.mp3")
        print(f"No shortcut, we recognized: {results}\n")

    finally:
        print("Cleaning up database...")
        djv.db.empty()
        print("Database cleanup complete.")

@app.function(
    image=image,
    gpu=gpu,
)
def init_gpu():
    print("Initializing GPU...")
    try:
        import cupy as cp
        print(f"CuPy version: {cp.__version__}")
        print(f"CuPy available devices: {cp.cuda.runtime.getDeviceCount()}")
        print(f"Current device: {cp.cuda.Device().id}")
        cp.cuda.Device(0).use()
        print("GPU device set successfully")
        return True
    except Exception as e:
        print(f"Error with CuPy: {e}")
        return False

@app.function(
    image=image,
    gpu=gpu,
)
def process_audio_files_modal():
    import sys
    import os
    import subprocess
    
    sys.path.insert(0, "/root")
    
    print("\nCPU Information:")
    subprocess.run("lscpu", shell=True)
    
    try:
        subprocess.run("cd /root/nostalgia && rm -f *.so && python3 setup.py clean --all && python3 setup.py build_ext --inplace", shell=True, check=True)

        print("Successfully built C++ extension")
    except subprocess.CalledProcessError as e:
        print(f"Failed to build C++ extension: {e}")
    
    gpu_initialized = init_gpu.remote()
    if not gpu_initialized:
        print("Failed to initialize GPU, continuing without GPU support")
    
    setup_postgres()
    
    from dejavu import Dejavu
    from dejavu.logic.recognizer.file_recognizer import FileRecognizer
    
    process_audio_files_local()

@app.local_entrypoint()
def main():
    process_audio_files_modal.remote()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true', help='Run locally instead of on Modal')
    args = parser.parse_args()

    if args.local:
        print("Running locally...")
        process_audio_files_local()
    else:
        print("Running on Modal...")
        main()

