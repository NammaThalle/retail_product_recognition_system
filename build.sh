./cleanup.sh
echo "======================================================================================================="
echo "                                     Preparing Dataset"
echo "======================================================================================================="
python model/src/data_preparation.py \
    --original-dataset-dir /home/nammathalle/work/retail_product_recognition_system/model/original_dataset \
    --new_dataset-dir /home/nammathalle/work/retail_product_recognition_system/model/dataset \
    --model-data-dir /home/nammathalle/work/retail_product_recognition_system/model/model_data \
    --templates-dir /home/nammathalle/work/retail_product_recognition_system/model/templates
echo "======================================================================================================="
echo ""
echo ""
echo "======================================================================================================="
echo "                                  Training the ResNet-50 model"
echo "======================================================================================================="
python model/src/ResNet-PyTorch/train.py \
    --dataset-dir /home/nammathalle/work/retail_product_recognition_system/model/dataset \
    --model-data-dir /home/nammathalle/work/retail_product_recognition_system/model/model_data
echo "======================================================================================================="
echo ""
echo ""
echo "======================================================================================================="
echo "                             Evaluating the trained ResNet-50 model"
echo "======================================================================================================="
python model/src/ResNet-PyTorch/test.py \
    --dataset-dir /home/nammathalle/work/retail_product_recognition_system/model/dataset \
    --model-data-dir /home/nammathalle/work/retail_product_recognition_system/model/model_data
echo "======================================================================================================="