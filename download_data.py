# download_data.py
import kagglehub

def main():
    print("Downloading Heart Disease dataset via kagglehub...")
    path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
    print("Dataset downloaded to:", path)
    return path

if __name__ == "__main__":
    main()
