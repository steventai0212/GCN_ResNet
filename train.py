import tensorflow as tf 
import numpy as np 
import os
from options import Option
from reconstruction_model import *
from data_loader import *
from utils import *
import argparse
from lib import graphh, coarsening, utils, mesh_sampling
from psbody.mesh import Mesh
from six.moves import xrange
import scipy.sparse as sp
from glob import glob
###############################################################################################
# training stage
###############################################################################################


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# training data and validation data
def parse_args():
    desc = "Deep3DFaceReconstruction"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--data_path', type=str, default='./celeba_processed_data', help='training data folder')
    parser.add_argument('--val_data_path', type=str, default='./celeba_val_processed_data', help='validation data folder')
    parser.add_argument('--model_name', type=str, default='./test1', help='model name')


    return parser.parse_args()

# initialize weights for resnet and facenet
def restore_weights_and_initialize(opt):
	var_list = tf.trainable_variables()
	g_list = tf.global_variables()
	# print(var_list)
	# exit()

	# add batch normalization params into trainable variables 
	bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
	bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
	var_list +=bn_moving_vars

	# create saver to save and restore weights
	resnet_vars = [v for v in var_list if 'resnet_v1_50' in v.name]
	facenet_vars = [v for v in var_list if 'InceptionResnetV1' in v.name]
	# resume_vars = [v for v in var_list if 'resnet_v1_50' in v.name or 'fc-' in v.name or 'gcn_decoder' in v.name]
	# d_vars = [var for var in var_list if var.name.startswith('discriminator')]

	# saver_resnet = tf.train.Saver(var_list = resnet_vars)
	# saver_facenet = tf.train.Saver(var_list = facenet_vars)
	# resume_net = tf.train.Saver(var_list = resume_vars)

	# saver_discriminator = tf.train.Saver(var_list=d_vars)
	saver_vars = [v for v in var_list if 'gcn' in v.name]
	# saver = tf.train.Saver(var_list = resnet_vars + [v for v in var_list if 'gcn_decoder' in v.name],max_to_keep = 50)
	saver = tf.train.Saver(var_list = saver_vars,max_to_keep = 50)

	# create session
	sess = tf.InteractiveSession(config = opt.config)

	# create summary op
	train_writer = tf.summary.FileWriter(opt.train_summary_path, sess.graph)
	val_writer = tf.summary.FileWriter(opt.val_summary_path, sess.graph)

	# initialization
	tf.global_variables_initializer().run()
	tf.local_variables_initializer().run()

	# saver_resnet.restore(sess,opt.R_net_weights)
	# saver_facenet.restore(sess,opt.Perceptual_net_weights)
	# if not opt.pretrain_stage:
	# 	resume_net.restore(sess,opt.pretrain_model)


	return saver, train_writer,val_writer, sess


# main function for training
def train():
	data_dir = './data'
	adj_path = os.path.join(data_dir, 'adjacency')
	ds_path = os.path.join(data_dir, 'downsamp_trans')
	us_path = os.path.join(data_dir, 'upsamp_trans')
	lap_path = os.path.join(data_dir, 'laplacians')

	if not os.path.isfile(lap_path + '0.npz'):
		print("Loading template data .. ")
		template_file_path = 'data/template.obj'
		template_mesh = Mesh(filename=template_file_path)
		ds_factors = [4,4,4,4]	# Sampling factor of the mesh at each stage of sampling
		print("Generating Transform Matrices ..")

		M,A1,D1,U1 = mesh_sampling.generate_transform_matrices(template_mesh, ds_factors)
		
		A = [x.astype('float32') for x in A1]
		D = [x.astype('float32') for x in D1]
		U = [x.astype('float32') for x in U1]

		print("Computing Graph Laplacians ..")
		L = [graphh.laplacian(a, normalized=True) for a in A]

		if not os.path.exists(data_dir):
			os.makedirs(data_dir)
		for i, a in enumerate(A):
			sp.save_npz(adj_path + '{}.npz'.format(i), a)
		for i, d in enumerate(D):
			sp.save_npz(ds_path + '{}.npz'.format(i), d)
		for i, u in enumerate(U):
			sp.save_npz(us_path + '{}.npz'.format(i), u)
		for i, l in enumerate(L):
			sp.save_npz(lap_path + '{}.npz'.format(i), l)
	else:
		print('Loading matrix for GCN')
		A = []
		D = []
		U = []
		L = []
		for a in sorted(glob('{}*.npz'.format(adj_path))):
			A.append(sp.load_npz(a))
		for d in sorted(glob('{}*.npz'.format(ds_path))):
			D.append(sp.load_npz(d))
		for u in sorted(glob('{}*.npz'.format(us_path))):
			U.append(sp.load_npz(u))
		for l in sorted(glob('{}*.npz'.format(lap_path))):
			L.append(sp.load_npz(l))

		
	p = [x.shape[0] for x in A]
	# set GCN parameter
	params = dict()
	# Building blocks.
	params['filter_']        = 'chebyshev5'
	params['brelu']          = 'b1relu'
	params['pool']           = 'poolwT'
	params['unpool']		 = 'poolwT'

	# Architecture.
	params['F_0']            = 3  # Number of graph input features.
	params['F']              = [16, 16, 16, 32]  # Number of graph convolutional filters.
	params['K']              = [6, 6, 6, 6]  # Polynomial orders.
	params['p']              = p #[4, 4, 4, 4]    # Pooling sizes.
	params['nz']             = [128]  # Output dimensionality of fully connected layers.
	params['L']              = L
	params['U']              = U
	params['D']              = D
	# params['M']              = M
	# Optimization.
	params['nv']             = 40000
	params['regularization'] = 5e-4
	params['dropout']        = 1


	# read BFM face model
	# transfer original BFM model to our model
	if not os.path.isfile('./BFM/BFM_model_front.mat'):
		transferBFM09()

	with tf.Graph().as_default() as graph:

		# training options
		args = parse_args()
		opt = Option(model_name=args.model_name)
		opt.data_path = [args.data_path]
		opt.val_data_path = [args.val_data_path]

		# load training data into queue
		train_iterator = load_dataset(opt)
		# create reconstruction model
		id_latent = tf.placeholder(tf.float32, [opt.batch_size, 3840], name='id_latent')
		pca_model = tf.placeholder(tf.float32, [opt.batch_size,35709,3], name='pca_model')
		model = Reconstruction_model(opt,graph,id_latent,pca_model,**params)
		
		# send training data to the model
		model.set_input(train_iterator)
		# update model variables with training data
		model.step(is_train = True)
		# summarize training statistics
		model.summarize()

		# several training stattistics to be saved
		train_stat = model.summary_stat
		train_img_stat = model.summary_img
		train_op = model.train_op
		photo_error = model.photo_loss
		lm_error = model.landmark_loss
		id_error = model.perceptual_loss
		shape_error = model.shape_loss
		norm_error = model.normal_loss
		# id_labels = model.id_labels
		id_emb = model.id_emb
		pca_facemodel = model.pca_facemodel
		

		# load validation data into queue
		val_iterator = load_dataset(opt,train=False)
		# send validation data to the model
		model.set_input(val_iterator)
		# only do foward pass without updating model variables
		model.step(is_train = False)
		# summarize validation statistics
		model.summarize()
		val_stat = model.summary_stat
		val_img_stat = model.summary_img

		# initialization
		saver, train_writer,val_writer, sess = restore_weights_and_initialize(opt)

		# freeze the graph to ensure no new op will be added during training
		sess.graph.finalize()

		# training loop
		# if opt.pretrain_stage:
		# 	iter_ = 0
		# 	print('starting training from iter 0')
		# else:
		# 	iter_ = opt.resume_iter
		# 	print('starting training from iter {}'.format(opt.resume_iter))
		# for i in xrange(iter_,opt.train_maxiter):
		for i in xrange(0,opt.train_maxiter):
			id_ = sess.run(id_emb)
			pca_ = sess.run(pca_facemodel)
			ffeed_dict = {id_latent:id_,pca_model:pca_}
			# _,ph_loss,lm_loss,id_loss,shape_loss,norm_loss = sess.run([train_op,photo_error,lm_error,id_error,shape_error,norm_error])
			_,ph_loss,lm_loss,id_loss,shape_loss,norm_loss = sess.run([train_op,photo_error,lm_error,id_error,shape_error,norm_error],feed_dict=ffeed_dict)

			print('Iter: %d; lm_loss: %f ; photo_loss: %f; id_loss: %f; shape_loss: %f; norm_loss: %f\n'%(i,np.sqrt(lm_loss),ph_loss,id_loss,shape_loss,norm_loss))

			# summarize training stats every <train_summary_iter> iterations
			if np.mod(i,opt.train_summary_iter) == 0:
				train_summary = sess.run(train_stat,feed_dict=ffeed_dict)
				train_writer.add_summary(train_summary,i)

			# summarize image stats every <image_summary_iter> iterations
			if np.mod(i,opt.image_summary_iter) == 0:
				train_img_summary = sess.run(train_img_stat,feed_dict=ffeed_dict)
				train_writer.add_summary(train_img_summary,i)

			# summarize validation stats every <val_summary_iter> iterations	
			if np.mod(i,opt.val_summary_iter) == 0:
				val_summary,val_img_summary = sess.run([val_stat,val_img_stat],feed_dict=ffeed_dict)
				val_writer.add_summary(val_summary,i)
				val_writer.add_summary(val_img_summary,i)

			# # save model variables every <save_iter> iterations	
			if np.mod(i,opt.save_iter) == 0:
				saver.save(sess,os.path.join(opt.model_save_path,'iter_%d.ckpt'%i))



if __name__ == '__main__':
	train()