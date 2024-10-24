# Retail Product Recognition System

## 1. Dataset Sources
- [Coke Classification Dataset](https://universe.roboflow.com/nandine-mrpzn/coke-classification-rbzab/dataset/4)
- [Soda Project Dataset](https://universe.roboflow.com/python-hrfbj/soda_project)
- [Snacks Dataset](https://huggingface.co/datasets/Matthijs/snacks/tree/main)
- [Dabur Active Juice Dataset](https://universe.roboflow.com/daburactivejuice/dabur_activejuice)

### Dataset Structure
<table>
  <tr>
    <td style="vertical-align: top;">
      <strong>Original Dataset</strong><br>
      <img src="metadata/original_dataset_structure.jpg" alt="Original Dataset" width="300">
    </td>
    <td style="vertical-align: top;">
      <strong>New Dataset</strong><br>
      <img src="metadata/new_dataset_structure.jpg" alt="New Dataset" width="300">
    </td>
  </tr>
</table>

## 2. Model Information
A **ResNet50** model is utilized to identify products across **15 retail products** within **3 categories**. This model processes the images provided by users through the frontend.

### Model Training Metrics
Below are the visualized examples of the model's training metrics:

<table>
  <tr>
    <td style="vertical-align: top;">
      <strong>Precision - 98.08%</strong><br>
      <img src="metadata/precision.jpg" alt="Precision" width="600">
    </td>
  </tr>
  <tr>
    <td style="vertical-align: top;">
      <strong>Recall - 98.02%</strong><br>
      <img src="metadata/recall.jpg" alt="Recall" width="600">
    </td>
  </tr>
  <tr>
    <td style="vertical-align: top;">
      <strong>F1 Score - 98.01%</strong><br>
      <img src="metadata/f1_score.jpg" alt="F1 Score" width="600">
    </td>
  </tr>
  <tr>
    <td style="vertical-align: top;">
      <strong>Train vs. Validation Loss</strong><br>
      <img src="metadata/train_vs_validation_loss.jpg" alt="Train vs. Validation Loss" width="600">
    </td>
  </tr>
</table>

## 3. Frontend
The frontend is developed using **ReactJS**, providing functionality to upload an image and receive the top **3 predictions** for that image. It includes the product category, exact product name, and classification confidence.

## 4. Backend
The backend is developed with **Python** and **FastAPI**. It initializes the model and exposes the prediction API to the frontend. When a user uploads an image, the frontend relays it to the backend via FastAPI, which handles product classification. Finally, the backend forwards the predictions back to the frontend.

## 5. API Reference
1. Frontend: [http://<machine_ip>:3000](http://<machine_ip>:3000)
2. Backend: [http://<machine_ip>:8000](http://<machine_ip>:8000)
3. Prediction Endpoint: [http://<machine_ip>:3000/predict/](http://<machine_ip>:3000/predict/) (POST) - Used through the frontend

## 6. Instructions to Run the Retail Product Recognition Application
1. Ensure you are using an **x86_64** machine.
2. Make sure you have **Docker** and **Docker Compose** installed.
3. Clone the repository with the following command:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
4. Build and run the application using Docker Compose:
    ```
    docker compose build
    docker compose up -d
    ```
5. Open your browser:
    * Frontend (React): http://<machine_ip>:3000 - **(IMPORTANT: Use CORS blocker extension for your browser)**
    * Backend (FastAPI): http://<machine_ip>:8000
6. Stopping the Project. To stop the project, run:
    ```
    docker-compose down
    ```

