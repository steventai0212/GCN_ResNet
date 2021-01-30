import tensorflow as tf 
from tensorflow.contrib.slim.nets import resnet_v1
slim = tf.contrib.slim
from inception_resnet_v1 import inception_resnet_v1
import scipy.sparse
import sklearn
import numpy as np
from lib import graphh


###############################################################################################
#Define R-Net and Perceptual-Net for 3D face reconstruction
###############################################################################################


def R_Net(inputs,is_training=True):
	#input: [Batchsize,H,W,C], 0-255, BGR image
	inputs = tf.cast(inputs,tf.float32)
	# standard ResNet50 backbone (without the last classfication FC layer)
	with slim.arg_scope(resnet_v1.resnet_arg_scope()):
		net,end_points = resnet_v1.resnet_v1_50(inputs,is_training = is_training ,reuse = tf.AUTO_REUSE)

	# Modified FC layer with 257 channels for reconstruction coefficients
	net_id = slim.conv2d(net, 80, [1, 1],
		activation_fn=None,
		normalizer_fn=None,
		weights_initializer = tf.zeros_initializer(),
		scope='fc-id')
	net_ex = slim.conv2d(net, 64, [1, 1],
		activation_fn=None,
		normalizer_fn=None,
		weights_initializer = tf.zeros_initializer(),
		scope='fc-ex')
	net_tex = slim.conv2d(net, 80, [1, 1],
		activation_fn=None,
		normalizer_fn=None,
		weights_initializer = tf.zeros_initializer(),
		scope='fc-tex')
	net_angles = slim.conv2d(net, 3, [1, 1],
		activation_fn=None,
		normalizer_fn=None,
		weights_initializer = tf.zeros_initializer(),
		scope='fc-angles')
	net_gamma = slim.conv2d(net, 27, [1, 1],
		activation_fn=None,
		normalizer_fn=None,
		weights_initializer = tf.zeros_initializer(),
		scope='fc-gamma')
	net_t_xy = slim.conv2d(net, 2, [1, 1],
		activation_fn=None,
		normalizer_fn=None,
		weights_initializer = tf.zeros_initializer(),
		scope='fc-XY')
	net_t_z = slim.conv2d(net, 1, [1, 1],
		activation_fn=None,
		normalizer_fn=None,
		weights_initializer = tf.zeros_initializer(),
		scope='fc-Z')

	net_id = tf.squeeze(net_id, [1,2], name='fc-id/squeezed')
	net_ex = tf.squeeze(net_ex, [1,2], name='fc-ex/squeezed')
	net_tex = tf.squeeze(net_tex, [1,2],name='fc-tex/squeezed')
	net_angles = tf.squeeze(net_angles,[1,2], name='fc-angles/squeezed')
	net_gamma = tf.squeeze(net_gamma,[1,2], name='fc-gamma/squeezed')
	net_t_xy = tf.squeeze(net_t_xy,[1,2], name='fc-XY/squeezed')
	net_t_z = tf.squeeze(net_t_z,[1,2], name='fc-Z/squeezed')

	net_ = tf.concat([net_id,net_ex,net_tex,net_angles,net_gamma,net_t_xy,net_t_z], axis = 1)

	return net_


def Perceptual_Net(input_imgs):
    #input_imgs: [Batchsize,H,W,C], 0-255, BGR image

    input_imgs = tf.reshape(input_imgs,[-1,224,224,3])
    input_imgs = tf.cast(input_imgs,tf.float32)
    input_imgs = tf.clip_by_value(input_imgs,0,255)
    input_imgs = (input_imgs - 127.5)/128.0

    #standard face-net backbone
    batch_norm_params = {
    'decay': 0.995,
    'epsilon': 0.001,
    'updates_collections': None}

    with slim.arg_scope([slim.conv2d, slim.fully_connected],weights_initializer=slim.initializers.xavier_initializer(), 
        weights_regularizer=slim.l2_regularizer(0.0),
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params):
        feature_128,_ = inception_resnet_v1(input_imgs, bottleneck_layer_size=128, is_training=False, reuse=tf.AUTO_REUSE)

    # output the last FC layer feature(before classification) as identity feature
    return feature_128

regularizers = []

def GCN(x, L, F, K, p, U,F_0):
	
	with tf.variable_scope('gcn_decoder', reuse=tf.AUTO_REUSE):
		N = x.get_shape()[0]
				#M, F, Fin = self.D[-1].shape[0], self.F[-1], self.F_0
		with tf.variable_scope('fc2'):
			x = fc(x, int(p[-1]*F[-1]))            # N x MF

		x = tf.reshape(x, [int(N), int(p[-1]), int(F[-1])])  # N x M x F

		for i in range(len(F)):
			with tf.variable_scope('upconv{}'.format(i+1)):
				with tf.name_scope('unpooling'):
					x = poolwT(x, U[-i-1])
				with tf.name_scope('filter'):
					x = chebyshev5(x, L[len(F)-i-1], F[-i-1], K[-i-1])
					print(L[-(i+1)], F[-(i+1)], K[-(i+1)])
				with tf.name_scope('bias_relu'):
					x = b1relu(x)

		with tf.name_scope('outputs'):
			x = chebyshev5(x, L[0], int(F_0),K[0])

	return x

def mesh_generator(image_emb, pca_mesh, D,L, F, K, p, U,F_0):

	with tf.variable_scope('gcn_mesh_generator', reuse=tf.AUTO_REUSE):
		decode_mesh = mesh_decoder(image_emb, L, K, p, U,F_0, reuse=tf.AUTO_REUSE)
		refine_mesh = mesh_refiner(pca_mesh, D,L, K, p, U,F_0 ,reuse=tf.AUTO_REUSE)

		with tf.variable_scope('gcn_mesh_concat'):
			concat = tf.concat([decode_mesh, refine_mesh], axis=-1)
			outputs = chebyshev5(concat, L[0], 3, 6)
		outputs = tf.nn.tanh(outputs)

	return outputs

def mesh_decoder(image_emb,L, K, p, U,F_0,reuse=tf.AUTO_REUSE):
	# if self.wide:
	# 	F = [32, 64, 128, 256]
	# else:
	F = [32, 16, 16, 16]
	N = image_emb.get_shape()[0]
	c_k = 6
	with tf.variable_scope('gcn_mesh_decoder', reuse=reuse):
		with tf.variable_scope('fc'):
			layer1 = fc(image_emb, p[-1] * F[0])  # N x MF
		layer1 = tf.reshape(layer1, [N, p[-1], F[0]])  # N x M x F

		with tf.variable_scope('resblock1'):
			with tf.name_scope('unpooling'):
				layer2 = poolwT(layer1, U[-1])
			layer2 = cheb_res_block(layer2, L[-2], F[1],c_k)
		with tf.variable_scope('resblock2'):
		# layer3 = tf.nn.dropout(layer2, 1 - self.drop_rate)
			with tf.name_scope('unpooling'):
				layer3 = poolwT(layer2, U[-2])
			layer3 = cheb_res_block(layer3, L[-3], F[2],c_k)
		with tf.variable_scope('resblock3'):
		# layer4 = tf.nn.dropout(layer3, 1 - self.drop_rate)
			with tf.name_scope('unpooling'):
				layer4 = poolwT(layer3, U[-3])
			layer4 = cheb_res_block(layer4, L[-4], F[3],c_k)
		with tf.variable_scope('resblock4'):
		# layer5 = tf.nn.dropout(layer4, 1 - self.drop_rate)
			with tf.name_scope('unpooling'):
				layer5 = poolwT(layer4, U[-4])
			outputs = cheb_res_block(layer5, L[-5], 3, c_k)
		#  relu=False)
		# outputs = tf.nn.tanh(outputs)
	return outputs

def mesh_refiner(pca_color,D,L, K, p, U,F_0, reuse=tf.AUTO_REUSE):
	# if self.wide:
	# 	F = [16, 32, 64, 128]
	# else:
	F = [16, 32, 32, 16]
	c_k = 6
	with tf.variable_scope('gcn_mesh_refiner', reuse=reuse):
		with tf.variable_scope('resblock1'):
			layer1 = cheb_res_block(pca_color, L[0], F[0],c_k)
		with tf.variable_scope('resblock2'):
			with tf.name_scope('pooling'):
				layer2 = poolwT(layer1, D[0])
			layer2 = cheb_res_block(layer2, L[1], F[1], c_k)
		with tf.variable_scope('resblock3'):
		# layer3 = tf.nn.dropout(layer2, 1 - self.drop_rate)
			layer3 = cheb_res_block(layer2, L[1], F[2], c_k)
		with tf.variable_scope('resblock4'):
	# layer4 = tf.nn.dropout(layer3, 1 - self.drop_rate)
			with tf.name_scope('unpooling'):
				layer4 = poolwT(layer3, U[0])
			layer4 = cheb_res_block(layer4, L[0], F[3], c_k)
			
		with tf.variable_scope('resblock5'):
	# layer5 = tf.nn.dropout(layer4, 1 - self.drop_rate)
			outputs = cheb_res_block(layer4, L[0], 3, c_k)
	#  relu=False)
	# outputs = tf.nn.tanh(outputs)
	return outputs
	
	

def _bias_variable(shape, regularization=True):
	initial = tf.constant_initializer(0.1)
	var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
	if regularization:
		regularizers.append(tf.nn.l2_loss(var))
	tf.summary.histogram(var.op.name, var)
	return var

def fc(x, Mout, relu=True):
	N, Min = x.get_shape()
	W = _weight_variable([int(Min), Mout], regularization=True)
	b = _bias_variable([Mout], regularization=True)
	x = tf.matmul(x, W) + b
	return tf.nn.relu(x) if relu else x

def poolwT( x, L):
	Mp = L.shape[0]
	N, M, Fin = x.get_shape()
	N, M, Fin = int(N), int(M), int(Fin)
	# Rescale transform Matrix L and store as a TF sparse tensor. Copy to not modify the shared L.
	L = scipy.sparse.csr_matrix(L)
	L = L.tocoo()
	indices = np.column_stack((L.row, L.col))
	L = tf.SparseTensor(indices, L.data, L.shape)
	L = tf.sparse_reorder(L)

	x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
	x = tf.reshape(x, [M, Fin*N])  # M x Fin*N
	x = tf.sparse_tensor_dense_matmul(L, x) # Mp x Fin*N
	x = tf.reshape(x, [Mp, Fin, N])  # Mp x Fin x N
	x = tf.transpose(x, perm=[2,0,1]) # N x Mp x Fin

	return x

def cheb_res_block(inputs, L, Fout, K, relu=True):
	_, _, Fin = inputs.get_shape().as_list()
	if Fin != Fout:
		with tf.variable_scope('shortcut'):
			shortcut = chebyshev5(inputs, L, Fout, 1)
	else:
		shortcut = inputs

	with tf.variable_scope('filter1'):
		x = chebyshev5(inputs, L, Fout, K)
	with tf.variable_scope('bias_relu1'):
		x = b1relu(x)

	with tf.variable_scope('filter2'):
		x = chebyshev5(x, L, Fout, K)
	x = tf.add(x, shortcut)
	if relu:
		with tf.variable_scope('bias_relu2'):
			x = b1relu(x)

	# with tf.variable_scope('filter3'):
	#   x = self.chebyshev5(x, L, 3, K)
	# if tanh:
	#   x = tf.nn.tanh(x)

	return x

def chebyshev5(x, L, Fout, K):
	N, M, Fin = x.get_shape()
	N, M, Fin = int(N), int(M), int(Fin)
	# Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
	L = scipy.sparse.csr_matrix(L)
	L = graphh.rescale_L(L, lmax=2)
	L = L.tocoo()
	indices = np.column_stack((L.row, L.col))
	L = tf.SparseTensor(indices, L.data, L.shape)
	L = tf.sparse_reorder(L)
	# Transform to Chebyshev basis
	x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
	x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
	x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
	def concat(x, x_):
		x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
		return tf.concat([x, x_], axis=0)  # K x M x Fin*N
	if K > 1:
		x1 = tf.sparse_tensor_dense_matmul(L, x0)
		x = concat(x, x1)
	for k in range(2, K):
		x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
		x = concat(x, x2)
		x0, x1 = x1, x2
	x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
	x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
	x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
	# Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
	W = _weight_variable([Fin*K, Fout], regularization=False)
	x = tf.matmul(x, W)  # N*M x Fout
	return tf.reshape(x, [N, M, Fout])  # N x M x Fout

def b1relu(x):
	"""Bias and ReLU. One bias per filter."""
	N, M, F = x.get_shape()
	b = _bias_variable([1, 1, int(F)], regularization=False)
	return tf.nn.relu(x + b)

def _weight_variable(shape, regularization=True):
	initial = tf.truncated_normal_initializer(0, 0.1)
	var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
	if regularization:
		regularizers.append(tf.nn.l2_loss(var))
	tf.summary.histogram(var.op.name, var)
	return var

	
def get_emb_coeff(net_name, inputs ,graph):
	with tf.gfile.GFile('/media/steven/3dface/deep3d_ResGCN/network/FaceReconModel.pb', 'rb') as f:
		face_rec_graph_def = tf.GraphDef()
		face_rec_graph_def.ParseFromString(f.read())
	# resized = tf.image.resize_images(inputs, [224, 224])
	# bgr_inputs = resized[..., ::-1]
	tf.import_graph_def(face_rec_graph_def, name=net_name, input_map={'input_imgs:0': inputs})
	image_emb = graph.get_tensor_by_name(net_name + '/resnet_v1_50/pool5:0')
	image_emb = tf.squeeze(image_emb, axis=[1, 2])
	coeff = graph.get_tensor_by_name(net_name + '/coeff:0')

	return image_emb, coeff

def get_img_feat(net_name, inputs,graph):
	with tf.gfile.GFile('/media/steven/3dface/deep3d_ResGCN/network/FaceNetModel.pb', 'rb') as f:
		face_net_graph_def = tf.GraphDef()
		face_net_graph_def.ParseFromString(f.read())
	# inputs should be in [0, 255]
	# facenet_input = tf.image.resize_image_with_crop_or_pad(inputs, 160, 160)
	# TODO: fix resize issue!!!
	facenet_input = tf.image.resize_images(inputs, [160, 160])

	facenet_input = (facenet_input - 127.5) / 128.0
	tf.import_graph_def(face_net_graph_def, name=net_name, input_map={'input:0': facenet_input,'phase_train:0': False})

	image_feat = graph.get_tensor_by_name(net_name + '/InceptionResnetV1/Logits/AvgPool_1a_8x8/AvgPool:0')

	image_feat = tf.squeeze(image_feat, axis=[1, 2])

	return image_feat