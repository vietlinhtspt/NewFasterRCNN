cd ..
cd tf-faster-rcnn/lib
make clean
make
cd ..
cd data
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make
cd ../../..