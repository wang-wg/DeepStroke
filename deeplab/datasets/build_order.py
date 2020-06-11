import glob
import math
import os.path
import sys
import build_data
import tensorflow as tf
import cv2
import numpy as np
import scipy.io as sio

FLAGS = tf.app.flags.FLAGS

_NUM_SHARDS = 4

def _convert_dataset(dataset_split, dataset_type_dir):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  print(dataset_type_dir)
  image_folder = dataset_type_dir+'JPEGImagesjpg'
  semantic_segmentation_folder = dataset_type_dir+'SegmentationClassAug'
  output_dir = dataset_type_dir + 'tfrecord-cross-strokeOrder-addLTHlabel_double'
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  maxMN=0
  strokeInformation=sio.loadmat('/home/wwg/data/DeepStroke/deeplab/datasets/strokeInformation'+'base'+'.mat')
  strokeNum=strokeInformation.get('strokeNumCharacter')
  strokeCount=strokeInformation.get('strokeCount')
#  print(strokeNum.shape)

  dataset = os.path.basename(dataset_split)[:-4]
  sys.stdout.write('Processing ' + dataset)
  filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
  num_images = len(filenames)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

  image_reader = build_data.ImageReader('jpeg', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        output_dir,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, len(filenames), shard_id))
        sys.stdout.flush()
        # Read the image.
        image_filename = os.path.join(
            image_folder, filenames[i] + '.' + FLAGS.image_format)
        if os.path.exists(image_filename)==False:
            continue
        image_data = tf.gfile.FastGFile(image_filename, 'r').read()
        height, width = image_reader.read_image_dims(image_data)
        # Read the stroke segmentation annotation.
        seg_filename = os.path.join(
            semantic_segmentation_folder,
            filenames[i] + '.' + FLAGS.label_format)
        seg_data = tf.gfile.FastGFile(seg_filename, 'r').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)

        semantic_segmentation_order_folder=dataset_type_dir+'OrderSegmentationClassAug'

        # Read the stroke order labeling annotation.
        seg_order_filename = os.path.join(
            semantic_segmentation_order_folder,
            filenames[i] + '.' + FLAGS.label_format)
        seg_data_order = tf.gfile.FastGFile(seg_order_filename, 'r').read()
        seg_height_order, seg_width_order = label_reader.read_image_dims(seg_data_order)
        # Add reference data for stroke segmentation.
        skel_filename = os.path.join(
              '/home/wwg/data/CCSSD/DATA_GB6763_LTH/LTH2017/OrderSegmentationClassAug',
              filenames[i] + '.' + FLAGS.label_format)
        skel_data = tf.gfile.FastGFile(skel_filename, 'r').read()

        # Add reference data for stroke order labeling.
        skel_35_filename = os.path.join(
              '/home/wwg/data/CCSSD/DATA_GB6763_LTH/LTH2017/SegmentationClassAug',
              filenames[i] + '.' + FLAGS.label_format)
        skel_35_data = tf.gfile.FastGFile(skel_35_filename, 'r').read()

        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        if height != seg_height_order or width != seg_width_order:
          raise RuntimeError('Shape mismatched between image and label_order.')

        if seg_height != seg_height_order or seg_width != seg_width_order:
          raise RuntimeError('Shape mismatched between label and label_order.')

        maxMN=max(height,width,seg_height,seg_width,seg_height_order,seg_width_order)

        filenameNum=int(filenames[i][2:filenames[i].index("_")])
        stroke_list_tmp=strokeNum[filenameNum-1,:]
        stroke_list=np.ones(35,dtype=int)
        stroke_list[1:35]=strokeNum[filenameNum-1,:]

        stroke_order_list=np.zeros(32,dtype=int)
        stroke_order_list[0:strokeCount[0,filenameNum-1]+1]=1
        example = build_data.image_seg_to_tfexample(
            image_data, filenames[i], height, width, seg_data,stroke_list,skel_data,skel_35_data, stroke_order_list, seg_data_order)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()
    print(maxMN)


def main(unused_argv):
  list_folder = '/home/wwg/data/CCSSD/'
  dataset_splits = glob.glob(os.path.join(list_folder, 'trainval.txt'))
  #dataset_type_all=['FZJTJW','FZJunHJW','LTH','FZMHJW','FZMiaoWuJW','FZMSTJW','FZKATJW']
  dataset_type_all=['FZLBJW','HLJ','SS']
  for dataset_type in dataset_type_all:
    dataset_type_dir='/home/wwg/data/CCSSD/DATA_GB6763_'+dataset_type+'/'+dataset_type+'2017/'
    #print(dataset_type_dir)
    for dataset_split in dataset_splits:
      _convert_dataset(dataset_split, dataset_type_dir)

if __name__ == '__main__':
  tf.app.run()
