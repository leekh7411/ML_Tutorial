import tensorflow as tf
import os
import urllib
import numpy
import gzip

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
def download_data(filename,work_directory):
    # Check the work directory and if not then make new directory
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)

    filepath = os.path.join(work_directory,filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.urlretrieve(SOURCE_URL+filename,filepath)
        statinfo  = os.stat(filepath)
        print ('Successfully downloaded', filename, statinfo.st_size,'bytes.')
    return filepath

def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4),dtype=dt)

def extract_images(filename):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file : %s' %
                (magic, filename)
            )
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf,dtype=numpy.uint8)
        data = data.reshape(num_images,rows,cols,1)
        return data


