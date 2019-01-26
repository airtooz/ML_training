import tensorflow as tf

a = tf.constant([2])
b = tf.constant([3])

c = tf.add(a,b)

session = tf.Session()

result = session.run(c)
print(result)

input('Press any key to continue')

hello = tf.constant('Hello world!')

print(session.run(hello).decode())

session.close()

### Avoid to close session everytime ###
#
#   with tf.Session() as session:
#       result = session.run(c)
#       print(result)
#
########################################


