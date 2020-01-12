cd ..
cd lib
make clean
make
cd ..

cd data
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make
cd ../../..

mkdir -p data/imagenet_weights
cd data/imagenet_weights
wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xzvf vgg_16_2016_08_28.tar.gz
mv vgg_16.ckpt vgg16.ckpt
cd ../..