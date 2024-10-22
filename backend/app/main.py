# app/main.py

from fastapi import FastAPI, UploadFile, File
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
import os

app = FastAPI()

# MongoDB connection setup
MONGO_DETAILS = "mongodb://localhost:27017"
client = AsyncIOMotorClient(MONGO_DETAILS)
database = client["product_db"]
product_collection = database["products"]

@app.get("/")
async def root():
    """
    This function serves as the root endpoint of the FastAPI application.

    Returns:
    dict: A dictionary containing a welcome message.
    """
    return {"message": "Retail Product Recognition Backend"}

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    """
    Uploads an image file, performs product recognition using a pre-trained model,
    and stores the recognition result in the MongoDB collection.

    Parameters:
    file (UploadFile): The image file to be uploaded. The default value is File(...), which means the file is required.

    Returns:
    dict: A dictionary containing the filename of the uploaded image and the product recognition result.
    """
    file_location = f"images/{file.filename}"
    
    # Save the uploaded image
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    # Call the model inference function here (you can load your pre-trained model)
    
    # Simulate product recognition result
    result = {"product_name": "Sample Product", "confidence": 0.95}

    # Insert recognition result into MongoDB
    await product_collection.insert_one(result)

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