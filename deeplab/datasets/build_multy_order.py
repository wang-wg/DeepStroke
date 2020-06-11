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

dataset_type='MULTY_ORDER'
dataset_type_dir='/home/wwg/data/CCSSD/DATA_GB6763_'+dataset_type+'/'+dataset_type+'2017/'

tf.app.flags.DEFINE_string(
    'list_folder',
    dataset_type_dir,
    'Folder containing lists for training and validation')
tf.app.flags.DEFINE_string(
    'output_dir',
    dataset_type_dir+'tfrecord-cross-strokeOrder-addLTHlabel_double',
    'Path to save converted SSTable of TensorFlow examples.')

_NUM_SHARDS = 4

dataset_type_all=['FZJTJW','FZJunHJW','LTH','FZMHJW','FZMiaoWuJW','FZMSTJW','FZKATJW','FZLBJW','HLJ','SS']

def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  #add strokeNum and so on imformation
  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)
  maxMN=0
  strokeInformation=sio.loadmat('/home/wwg/data/DeepStroke/deeplab/datasets/strokeInformationbase.mat')
  strokeInformationLTH=sio.loadmat('/home/wwg/data/DeepStroke/deeplab/datasets/strokeInformation'+'LTH'+'.mat')
  strokeNum=strokeInformation.get('strokeNumCharacter')
  strokeNumLTH=strokeInformationLTH.get('strokeNumCharacter')
  strokeCount=strokeInformation.get('strokeCount')
  strokeCountLTH=strokeInformationLTH.get('strokeCount')
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
        FLAGS.output_dir,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, len(filenames), shard_id))
        sys.stdout.flush()
        # Read the image.
        for j in range(7):
          dataset_type_tmp=dataset_type_all[j]
          dataset_type_dir='/home/wwg/data/CCSSD/DATA_GB6763_'+dataset_type_tmp+'/'+dataset_type_tmp+'2017/'
          image_folder=dataset_type_dir+'JPEGImagesjpg'
          semantic_segmentation_folder=dataset_type_dir+'SegmentationClassAug'
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

          # Read the stroke order labeling annotation.
          semantic_segmentation_order_folder=dataset_type_dir+'OrderSegmentationClassAug'
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
          stroke_list=np.ones(35,dtype=int)
          stroke_list[1:35]=strokeNum[filenameNum-1,:]
          if dataset_type_tmp=='LTH':
            stroke_list[1:35]=strokeNumLTH[filenameNum-1,:]

          stroke_order_list=np.zeros(32,dtype=int)
          stroke_order_list[0:strokeCount[0,filenameNum-1]+1]=1
          if dataset_type_tmp=='LTH':
            stroke_order_list[0:strokeCountLTH[0,filenameNum-1]+1]=1
          if i<6763:
            example = build_data.image_seg_to_tfexample(
                image_data, dataset_type_tmp+filenames[i], height, width, seg_data,stroke_list,skel_data, skel_35_data,stroke_order_list, seg_data_order)
          else:
            example = build_data.image_seg_to_tfexample(
                image_data, dataset_type_tmp+'Add'+filenames[i], height, width, seg_data,stroke_list,skel_data, skel_35_data, stroke_order_list, seg_data_order)
          tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()
    print(maxMN)


def main(unused_argv):
  dataset_splits = glob.glob(os.path.join(FLAGS.list_folder, '*.txt'))
  for dataset_split in dataset_splits:
    _convert_dataset(dataset_split)


if __name__ == '__main__':
  tf.app.run()
