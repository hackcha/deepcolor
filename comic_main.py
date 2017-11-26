from dc_main import *


def comic_it(c, files1, index, img_size = 512):
    batch = np.array([cv2.resize(imread(batch_file), (img_size, img_size)) for batch_file in files1])
    batch_normalized = batch / 255.0
    batch_edge = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                 cv2.THRESH_BINARY, blockSize=9, C=2) for ba in batch]) / 255.0
    batch_edge = np.expand_dims(batch_edge, 3)
    batch_colors = np.array([c.imageblur(ba, True) for ba in batch]) / 255.0

    recreation = c.sess.run(c.generated_images,
                            feed_dict={c.real_images: batch_normalized, c.line_images: batch_edge,
                                       c.color_images: batch_colors})
    out_file = 'data/cmerge_{}.jpg'.format(index)
    edge_file = 'data/cedge_{}.jpg'.format(i)
    ims(edge_file, merge_color(batch_edge, [c.batch_size_sqrt, c.batch_size_sqrt]))
    ims(out_file, merge_color(recreation, [c.batch_size_sqrt, c.batch_size_sqrt]))


if __name__ == '__main__':
    img_size= 512
    import dc_main
    dc_main.upsplash = False
    c = Color(img_size, 1)
    c.loadmodel( False )
    print('load model done.')
    files1 = ['data/timg_.jpg','data/timg.jpg', 'data/lyf1.jpg', 'data/lyf2.jpg']
    for i, file1 in enumerate( files1):
        comic_it(c,[file1], i, img_size=img_size)
    print('done.')
