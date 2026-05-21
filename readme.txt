About:
    - Create a deep dreaming using DNN.
    - Technique to visualize what individual layers's activities represent.
    - This give a new take on gradient descent and new ways of using keras.

Running code:
1. Create a virutal environment
    - python -m venv myvenv
2. Download all the required packages
    - pip install -r requirements.txt
3. Run the code using the command
    - python dream_dnn.py <image> <X> <N> 
        image - image path
        X - layer selected
        N - Number of iterations
    eg: python dream_dnn.py img.png 56 400


Output:
1. So it will take the layer 56 from the pretrained mobilenet model.
2. And it will produce patterns. 
3. The image generated at every 10th iteration is stored in the output directory
4. The final image after 400 iterations will be displayed.