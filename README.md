# CSE382M_final
Final project for the introduction to Machine Learning CSE382M course.
This project aims to analyze various dimention reduction techniques on the image dataset. The dataset is available for download using the [following link](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition?resource=download) 


To install all of the necessary packages you can use (don't reccomend so far, but you may try and install a lot of packages that you may not need):

```
pip install -r requirements.txt
```

To make all of the images of one size, for the further work, use `scripts/resize.ipynb` script.


[]()
## TODO:
 1. In the `scripts/dim_reduction_evaluation.ipynb` script, add SVD dimention reduction and compare it to the existing distance evaluation figure plotting.
  

2. Test and validate the LJ lemma, kernel case:
    - Use polynomial kernell and do the same as for vanila LJ
    - Use exponential Kernel and do the same as for vanila LJ

3. Train CNN architecture to reach good classification accucracy (at least 95%) on the test dataset. (pretrained NN can be used here) (Ill be using `scripts/recogniiton_CNN.ipynb` for this) (More or less done)

    - Test its performance with similar size NN (multilayer perceptron), compare, make a nice figures out of it.
4. Train encoder-decoder CNNs using adversarial NN training