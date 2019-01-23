import cv2
#import visualize_new

def mkarr(i, j):
  return (i,j)
def save(arr, path):
  count = arr.shape[0]
  arr = arr * 127.5 + 127.5
  for i in range(count):
    cv2.imwrite(path  + str(i) + ".png", arr[i])

def save_detected(results, save_path):
  r = results[0]
  visualize_new.display_instances(save_path, image, r['rois'], r['masks'], r['class_ids'], 
                             0, r['scores'])

def make_list(img):
  return [img]