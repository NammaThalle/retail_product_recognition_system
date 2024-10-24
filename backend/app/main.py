from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
from starlette.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import uvicorn
from resnet.inference import ProductModel

app = FastAPI()
# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

product_classifier = ProductModel()


@app.get("/")
async def root():
    """
    This function serves as the root endpoint of the FastAPI application.

    Returns:
    dict: A dictionary containing a welcome message.
    """
    return {"message": "Retail Product Recognition Backend"}

@app.post("/predict/")
async def upload_image(file: UploadFile = File(...)):
    """
    Uploads an image file, performs product recognition using a pre-trained model,
    and stores the recognition result in the MongoDB collection.

    Parameters:
    file (UploadFile): The image file to be uploaded. The default value is File(...), which means the file is required.

    Returns:
    dict: A dictionary containing the filename of the uploaded image and the product recognition result.
    """
    # Read the file content and convert it to a numpy array
    file_bytes = np.frombuffer(file.file.read(), np.uint8)

    # Decode the numpy array into an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Run model inference in a thread pool
    result = await run_in_threadpool(product_classifier.classifyProducts, image)
    # Optionally delete the uploaded image after inference
    # os.remove(file_location)

    # Insert recognition result into MongoDB
    # await product_collection.insert_one(result)

    return {"filename": file.filename, "result": result}

@app.post("/add-product/")
async def add_product(product_name: str, price: float):
    """
    This function adds a new product to the MongoDB collection.

    Parameters:
    product_name (str): The name of the product.
    price (float): The price of the product.

    Returns:
    dict: A dictionary containing a success message.
    """
    product = {"name": product_name, "price": price}
    await product_collection.insert_one(product)
    return {"message": "Product added successfully"}

if __name__ == "__main__":
    config = uvicorn.Config(app, host='0.0.0.0', port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()

    # MongoDB connection setup
    MONGO_DETAILS = "mongodb://localhost:27017"
    client = AsyncIOMotorClient(MONGO_DETAILS)
    database = client["product_db"]
    product_collection = database["products"]