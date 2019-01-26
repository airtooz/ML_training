import tensorflow as tf

one = tf.constant(1)
state = tf.Variable(0)

### What if this script of code ###
#   
#   state = tf.assign(state, 0)
#   
#   sess.run(state) will call the assign everytime therefore variable state will always be 0
#
###################################

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

### Variables must be initialized by running an initialization operation after having launched the graph.
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))  ### print out initial value for state
    for _ in range(5):
        sess.run(update)    ### state = new_value
        print(sess.run(state))

