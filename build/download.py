from huggingface_hub import hf_hub_download

# Download the pytorch model
hf_hub_download(repo_id="AiresPucrs/LeNNon-Smile-Detector",
                filename="LeNNon-Smile-Detector.pt",
                local_dir="./model",
                repo_type="model"
                )

# Download the source implementation of the model's architecture
hf_hub_download(repo_id="AiresPucrs/LeNNon-Smile-Detector",
                filename="lennon.py",
                local_dir="./model",
                repo_type="model"
                )