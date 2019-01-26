import tensorflow as tf
import numpy as np

# Placeholders are like holes in the program
# Important: This tensor will produce an error if evaluated. Its value must be fed using the feed_dict.
a = tf.placeholder(tf.float32)
b = a*2

# array can also be fed in as numpy array
arr = np.array([[1.2,2.2,3.3],[1.4,0.8,0.2],[1.6,2.4,4.3]])

# the placeholder can be fed in multideminsional array
dictionary = {a:arr}
with tf.Session() as sess:
    result = sess.run(b, feed_dict = dictionary)
    print(result)
