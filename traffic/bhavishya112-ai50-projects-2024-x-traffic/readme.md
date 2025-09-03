My Experiences - bhavishya sharma

-batch_size 
    * reducing batch size makes the model train slower, and vice versa
    * but increasing batch size too much worsens accuracy and loss

-image filters
    * increasing the number of image filters increased the time/step, so eventually total training time
    * but doing so improved the accuracy and loss to heavenly amounts
    * took like 400ms/step at 500 image filters 

-image convolution + pooling
    * since training images were very small [30px x 30px], adding more image convolution + pooling pairs didn't help.
    * after 2 pairs, it decreased accuracy by 1%-2% 

-for output layer
    * using relu for output layer just downed my accuracy to 0.005 and loss became nan (im so silly)
    * so use softmax : classifies multiple labels   into their probability,
        sigmoid : classifies 2 labels as 1 or 0
    
