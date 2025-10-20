# Market Signal Forecasting - Azure Container Apps Deployment

This repository demonstrates how to containerize and deploy a Python/Flask-based forecasting model (Prophet + LSTM) for stock sentiment and price prediction using **Azure Container Apps**.

The deployment was tested successfully using an **amd64 Docker image**.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Local Setup](#local-setup)
3. [Architecture Diagram](#architecture-diagram) 
4. [Docker Build](#docker-build)  
5. [Azure Container Apps Deployment](#azure-container-apps-deployment)  
6. [Testing the Endpoint](#testing-the-endpoint)  
7. [Problems & Solutions](#problems--solutions)
8. [Notes](#notes)

---

## Project Overview

- **Frameworks:** Flask, TensorFlow, PyTorch, scikit-learn, Prophet  
- **Docker Architecture:** amd64 (linux/amd64)  
- **Cloud Platform:** Azure Container Apps  
- **Purpose:** Serve a REST API for stock signal forecasting based on sentiment and price data.

---

## Local Setup

1. Clone the repository:
  ```bash
   git clone <repo-url>
   cd Market_Signal_Forecasting
  ```

2. Install Dependencies Locally:
   - look through requirements.txt as well as Dockerfile to figure out what to install for the env.

3. Test the Flask app locally:
   ```bash
   python serve.py
   ```

___

## Architecture Diagram

```mermaid
graph TD
    A["Local MacBook"] -->|Build Docker Image| B["Docker Image amd64"]
    B -->|Push| C["Azure Container Registry (ACR)"]
    C -->|Deploy| D["Azure Container App Environment"]
    D -->|Run Container| E["Container App: Flask API serve.py"]
    E -->|Expose Port 8080| F["External Ingress / Public Endpoint"]
    F -->|Invoke via curl| G["Receive Predictions JSON"]
```

---

## Docker Build

1. Build the amd64 Docker image:
   ```bash
   docker buildx build --platform linux/amd64 -t market-signal-forecasting:latest .
   ```

2. Run the container locally for testing (**Note** -> this will not work if running from macOS, you'd need to build the docker image specifically for arm64):
   ```bash
   docker run -p 8080:8080 market-signal-forecasting:latest
   ````

3. Verify Endpoint:
   ```bash
   curl -X POST http://localhost:8080/invocations \
      -H "Content-Type: application/json" \
      -d '{"company": "Apple", "symbol": "AAPL"}'
   ```

---

## Azure Container Apps Deployment

1. Login to Azure CLI and Azure Container Registry
   ```bash
   az login

   az group create --name msf-centralus-rg --location centralus

   az acr create \
      --resource-group msf-centralus-rg \                                
      --name acrmarketsignal \
      --sku Basic \                                  
      --location centralus \
      --admin-enabled true

   az acr login --name acrmarketsignal --resource-group msf-centralus-rg
   ```

2. Tag and push Docker image to ACR:
   ```bash
   docker tag market-signal-forecasting:latest acrmarketsignal.azurecr.io/market-signal-forecasting:latest
   docker push acrmarketsignal.azurecr.io/market-signal-forecasting:latest
   ```

3. Create Container App Environment:
   ```bash
   az containerapp env create \
      --name market-signal-env \
      --resource-group msf-centralus-rg \
      --location centralus
   ```

4. Deploy Container App:
   ```bash
   az containerapp create \
      --name market-signal-forecasting \
      --resource-group msf-centralus-rg \
      --environment market-signal-env \
      --image acrmarketsignal.azurecr.io/market-signal-forecasting:latest \
      --cpu 1 \
      --memory 2Gi \
      --ingress external \
      --target-port 8080 \
      --registry-server acrmarketsignal.azurecr.io \
      --registry-username acrmarketsignal \
      --registry-password <ACR_PASSWORD>
   ```

5. Retrieve the public endpoint URL:
   ```bash
   az containerapp show \
      --name market-signal-forecasting \
      --resource-group msf-centralus-rg \
      --query properties.configuration.ingress.fqdn
   ```

---

## Testing the Endpoint
- Use the **public URL** provided by Azure Container Apps with a POST request:
   ```bash
   curl -X POST https://<public-fqdn>/invocations \
      -H "Content-Type: application/json" \
      -d '{"company": "Apple", "symbol": "AAPL"}'
   ```
- Expected response includes RMSE values for Prophet and LSTM predictions.

![Testing predictability in Azure Console Shell and local terminal](images/MSFAzureContainerApp.png)

___

## Problems And Solutions

| Problem                                   | Root Cause                                         | Solution                                                   |
|-------------------------------------------|--------------------------------------------------|------------------------------------------------------------|
| OCI runtime create failed locally          | CMD wasn’t pointing to Python interpreter correctly | Changed `CMD ["python", "serve.py"]` in Dockerfile       |
| curl: (7) Failed to connect to localhost:8080 | Container wasn’t publishing the internal port   | Added `-p 8080:8080` when testing locally                |
| Azure Container Apps rejected ARM64 image | Container App only supports linux/amd64          | Rebuilt Docker image for amd64 and pushed to ACR         |
| Azure environment missing                  | Tried deploying to non-existent container app environment | Created environment using `az containerapp env create` |
| TensorFlow/Apple Silicon errors locally   | Incompatibility between TensorFlow-CPU and Apple Silicon | Installed `tensorflow==2.16.1` with proper platform compatibility |


___

## Notes

- All deployments were performed using Azure Container Apps with `amd64 architecture`.
- Local `ARM64` builds can run on Apple Silicon, but Azure Container Apps requires amd64.
- Endpoint testing should always be performed using the public `FQDN` of the container app.


