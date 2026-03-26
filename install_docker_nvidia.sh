#!/bin/bash
set -e

echo "=== Installing Docker ==="
curl -fsSL https://get.docker.com | sh

echo "=== Adding current user to docker group ==="
usermod -aG docker $SUDO_USER 2>/dev/null || true

echo "=== Installing nvidia-container-toolkit ==="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit

echo "=== Configuring Docker to use NVIDIA runtime ==="
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "=== Verifying NVIDIA Docker access ==="
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi

echo ""
echo "Done. Docker + nvidia-container-toolkit installed and verified."
echo "You may need to log out and back in for the docker group to take effect."
