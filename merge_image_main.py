from dc_main import *


def merge_it(c, files1, files2, index, img_size = 512):
    batch = np.array([cv2.resize(imread(batch_file), (img_size, img_size)) for batch_file in files1])
    batch_c = np.array([cv2.resize(imread(batch_file), (img_size, img_size)) for batch_file in files2])
    batch_normalized = batch / 255.0
    batch_edge = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                 cv2.THRESH_BINARY, blockSize=9, C=2) for ba in batch]) / 255.0
    batch_edge = np.expand_dims(batch_edge, 3)
    batch_colors = np.array([c.imageblur(ba, True) for ba in batch_c]) / 255.0

    recreation = c.sess.run(c.generated_images,
                            feed_dict={c.real_images: batch_normalized, c.line_images: batch_edge,
                                       c.color_images: batch_colors})
    out_file = 'data/lmerge_{}.jpg'.format(index)
    edge_file = 'data/ledge_{}.jpg'.format(index)
    ims(edge_file, merge_color(batch_edge, [c.batch_size_sqrt, c.batch_size_sqrt]))
    ims(out_file, merge_color(recreation, [c.batch_size_sqrt, c.batch_size_sqrt]))


if __name__ == '__main__':
    img_size= 256
    c = Color(img_size, 1)
    c.loadmodel( False )
    print('load model done.')
    # files1 = ['data/1.jpg','data/2.jpg','data/3.jpg','data/4.jpg','data/1.jpg','data/2.jpg','data/3.jpg','data/4.jpg']
    # # files2 = ['data/lm.jpg','data/lm.jpg','data/lm.jpg','data/lm.jpg','data/cm.jpg','data/cm.jpg','data/cm.jpg','data/cm.jpg']
    # files2 = ['data/m_m.jpg', 'data/m_m.jpg', 'data/m_m.jpg', 'data/m_m.jpg', 'data/wm_m.jpg', 'data/wm_m.jpg', 'data/wm_m.jpg',
    #           'data/wm_m.jpg']
    # files1 = ['data/h1.jpg','data/1.jpg']
    # files2 = ['data/h2.jpg', 'data/h2.jpg']
    files1 = ['data/timg_.jpg', 'data/lyf1.jpg']
    files2 = ['data/timg.jpg', 'data/lyf2.jpg']
    for i, (file1, file2) in enumerate( zip(files1, files2)):
        merge_it(c,[file1], [file2], i, img_size=img_size)
    print('done.')
