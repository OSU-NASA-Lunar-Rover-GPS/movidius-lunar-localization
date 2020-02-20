import tensorflow as tf
from tensorflow.python.framework import graph_io





saver = tf.compat.v1.train.import_meta_graph('./tf_model_v1/model.ckpt-0.meta', clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess,tf.compat.v1.train.latest_checkpoint('./tf_model_v1/'))

output_node_names = "softmax_tensor"
output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
    sess,  # The session
    input_graph_def,  # input_graph_def is useful for retrieving the nodes
    output_node_names.split(",")
)

output_graph = "model.pb"
frozen = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["softmax_tensor"])
graph_io.write_graph(frozen, './tf_model_v1', output_graph, as_text=False)
# with tf.gfile.GFile(output_graph, "wb") as f:
#     f.write(output_graph_def.SerializeToString())
sess.close()